"""
Gluon MOE sum-reduce.

Replaces sglang's ``_moe_sum_reduce_kernel`` for the bf16 path. Sums an
intermediate cache of shape ``(M, topk, D)`` along the topk axis, optionally
multiplying by a ``routed_scaling_factor``. Output is ``(M, D)``.
"""

from __future__ import annotations

import os

import torch
import triton

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl

    _GLUON_AVAILABLE = True
except ImportError:
    _GLUON_AVAILABLE = False
    gluon = None
    gl = None


_USE_GLUON_REDUCE_ENV = "SGLANG_USE_GLUON_REDUCE"


def use_gluon_reduce() -> bool:
    if not _GLUON_AVAILABLE:
        return False
    val = os.environ.get(_USE_GLUON_REDUCE_ENV, "0").lower()
    return val in ("1", "true", "yes", "on")


if _GLUON_AVAILABLE:

    @gluon.jit
    def gluon_moe_sum_reduce_kernel(
        in_ptr,                      # (M, topk, D) bf16
        in_stride_m,
        in_stride_topk,
        out_ptr,                     # (M, D) bf16
        out_stride_m,
        token_num,                   # M
        topk_num,                    # topk
        hidden_dim,                  # D
        routed_scaling_factor: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_D: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        token_block_id = gl.program_id(0)
        dim_block_id = gl.program_id(1)

        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[8, 8],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        )

        offs_m = token_block_id * BLOCK_M + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, layout)
        )
        offs_d = dim_block_id * BLOCK_D + gl.arange(
            0, BLOCK_D, layout=gl.SliceLayout(0, layout)
        )

        mask_m = offs_m < token_num
        mask_d = offs_d < hidden_dim
        mask_2d = mask_m[:, None] & mask_d[None, :]

        in_stride_m_64 = in_stride_m.to(gl.int64)
        in_stride_topk_64 = in_stride_topk.to(gl.int64)
        out_stride_m_64 = out_stride_m.to(gl.int64)

        base_ptrs = (
            in_ptr
            + offs_m[:, None].to(gl.int64) * in_stride_m_64
            + offs_d[None, :].to(gl.int64)
        )

        acc = gl.zeros((BLOCK_M, BLOCK_D), dtype=gl.float32, layout=layout)
        for i in range(0, topk_num):
            tile = gl.load(
                base_ptrs + i * in_stride_topk_64,
                mask=mask_2d,
                other=0.0,
            )
            acc = acc + tile.to(gl.float32)
        acc = acc * routed_scaling_factor

        store_ptrs = (
            out_ptr
            + offs_m[:, None].to(gl.int64) * out_stride_m_64
            + offs_d[None, :].to(gl.int64)
        )
        gl.store(store_ptrs, acc.to(out_ptr.type.element_ty), mask=mask_2d)


def invoke_gluon_moe_sum_reduce(
    input_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    routed_scaling_factor: float = 1.0,
    block_m: int = 1,
    block_d: int = 1024,
    num_warps: int = 4,
) -> None:
    """Sum ``input[M, topk, D]`` along ``topk`` into ``output[M, D]``.

    Matches the public signature of ``sgl_kernel.moe_sum_reduce``.
    """
    assert _GLUON_AVAILABLE, "Gluon is not available in this Triton build"
    assert input_tensor.dim() == 3
    assert output_tensor.dim() == 2
    assert input_tensor.is_contiguous()
    assert output_tensor.is_contiguous()

    M, topk, D = input_tensor.shape
    assert output_tensor.shape == (M, D)

    grid = (triton.cdiv(M, block_m), triton.cdiv(D, block_d))

    gluon_moe_sum_reduce_kernel[grid](
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        output_tensor,
        output_tensor.stride(0),
        M,
        topk,
        D,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
