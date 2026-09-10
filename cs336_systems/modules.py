import torch

import torch
import triton
import triton.language as tl


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,

    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,

    N_QUERIES, N_KEYS,
    scale,

    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
):

    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    q = tl.load(Q_block_ptr)

    maxnum = tl.full(
        (Q_TILE_SIZE,),
        -float("inf"),
        tl.float32,
    )

    l = tl.zeros(
        (Q_TILE_SIZE,),
        tl.float32,
    )

    answer = tl.zeros(
        (Q_TILE_SIZE, D),
        tl.float32,
    )

    for j in range(0, N_KEYS, K_TILE_SIZE):

        k = tl.load(K_block_ptr)
        v = tl.load(V_block_ptr)

        score = tl.dot(q, k.T) * scale

        block_max = tl.max(score, axis=1)
        new_max = tl.maximum(maxnum, block_max)

        alpha = tl.exp(maxnum - new_max)

        p = tl.exp(
            score - new_max[:, None]
        )

        answer = answer * alpha[:, None]

        p_for_v = p.to(V_block_ptr.type.element_ty)

        answer = tl.dot(
            p_for_v,
            v,
            acc=answer,
        )

        l = (
            l * alpha
            + tl.sum(p, axis=1)
        )

        maxnum = new_max

        K_block_ptr = tl.advance(
            K_block_ptr,
            (K_TILE_SIZE, 0),
        )

        V_block_ptr = tl.advance(
            V_block_ptr,
            (K_TILE_SIZE, 0),
        )

    answer = answer / l[:, None]

    L = maxnum + tl.log(l)

    tl.store(
        O_block_ptr,
        answer.to(O_block_ptr.type.element_ty),
    )

    tl.store(
        L_block_ptr,
        L,
    )

class flashattention_autograd_function_pytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        size = 16

        q = Q.reshape(-1, Q.shape[-2], Q.shape[-1])
        k = K.reshape(-1, K.shape[-2], K.shape[-1])
        v = V.reshape(-1, V.shape[-2], V.shape[-1])

        maxnum = torch.full(
            (q.shape[0], q.shape[1]),
            float("-inf"),
            device=Q.device,
            dtype=torch.float32,
        )

        l = torch.zeros(
            (q.shape[0], q.shape[1]),
            device=Q.device,
            dtype=torch.float32,
        )

        answer = torch.zeros(
            (q.shape[0], q.shape[1], V.shape[-1]),
            device=Q.device,
            dtype=torch.float32,
        )

        L = torch.empty(
            (q.shape[0], q.shape[1]),
            device=Q.device,
            dtype=torch.float32,
        )

        scale = Q.shape[-1] ** -0.5

        for i in range(0, q.shape[1], size):
            q_block = q[:, i:i + size, :]

            for j in range(0, k.shape[1], size):
                k_block = k[:, j:j + size, :]
                v_block = v[:, j:j + size, :]

                score = (
                    q_block.float()
                    @ k_block.float().transpose(-2, -1)
                ) * scale

                old_max = maxnum[:, i:i + size]
                block_max = torch.max(score, dim=-1).values
                new_max = torch.maximum(old_max, block_max)

                alpha = torch.exp(old_max - new_max)
                p = torch.exp(score - new_max[:, :, None])

                answer[:, i:i + size, :] = (
                    answer[:, i:i + size, :] * alpha[:, :, None]
                    + p @ v_block.float()
                )

                l[:, i:i + size] = (
                    l[:, i:i + size] * alpha
                    + p.sum(dim=-1)
                )

                maxnum[:, i:i + size] = new_max

            answer[:, i:i + size, :] /= l[:, i:i + size, None]

            L[:, i:i + size] = (
                maxnum[:, i:i + size]
                + torch.log(l[:, i:i + size])
            )

        O = answer.to(Q.dtype).reshape(Q.shape)
        L = L.reshape(Q.shape[:-1])

        ctx.save_for_backward(L, Q, K, V, O)
        return O
    
class flashattention_autograd_function_triton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):

        batch_size = Q.shape[0]
        N_QUERIES = Q.shape[-2]
        N_KEYS = K.shape[-2]
        D = Q.shape[-1]

        O = torch.empty_like(Q)

        L = torch.empty(
            Q.shape[:-1],
            device=Q.device,
            dtype=torch.float32,
        )

        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16

        scale = D ** -0.5

        T_q = triton.cdiv(N_QUERIES, Q_TILE_SIZE)

        grid = (T_q, batch_size)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,

            Q.stride(0), Q.stride(-2), Q.stride(-1),
            K.stride(0), K.stride(-2), K.stride(-1),
            V.stride(0), V.stride(-2), V.stride(-1),
            O.stride(0), O.stride(-2), O.stride(-1),
            L.stride(0), L.stride(-1),

            N_QUERIES,
            N_KEYS,
            scale,

            D=D,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
        )

        ctx.save_for_backward(L, Q, K, V, O)

        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError