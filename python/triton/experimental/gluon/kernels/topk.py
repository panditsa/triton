"""
Gluon top-k + softmax for MOE routing.

Replaces the ``topk_softmax`` C++ kernel from sgl_kernel for the bf16/fp16/fp32
gating-output path. One CTA per token. The kernel:

    1. Loads ``logits[m, :]`` of shape ``(E,)``.
    2. Computes a numerically-stable softmax in fp32.
    3. Iteratively extracts the top-k: max-reduce, find the lowest index where
       the max occurs, write it out, set that slot to ``-inf``, repeat ``top_k``
       times.
    4. Optionally renormalizes the top-k weights to sum to 1.

The iterative top-k is O(top_k * log E) per token (the reductions are
parallel). For E in {8, 60, 256} and top_k in {2, 4, 8} this is well below
the cost of the surrounding GEMMs.
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


_USE_GLUON_TOPK_ENV = "SGLANG_USE_GLUON_TOPK"


def use_gluon_topk() -> bool:
    if not _GLUON_AVAILABLE:
        return False
    val = os.environ.get(_USE_GLUON_TOPK_ENV, "0").lower()
    return val in ("1", "true", "yes", "on")


if _GLUON_AVAILABLE:

    @gluon.jit
    def gluon_topk_softmax_kernel(
        logits_ptr,           # (M, E)
        out_weights_ptr,      # (M, top_k) fp32
        out_ids_ptr,          # (M, top_k) int32
        logits_stride_m,
        weights_stride_m,
        ids_stride_m,
        E,                    # runtime number of experts (pass as int, masked)
        TOP_K: gl.constexpr,
        BLOCK_E: gl.constexpr,         # next_power_of_2(E), upper bound
        RENORMALIZE: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        pid = gl.program_id(0)

        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[max(1, BLOCK_E // (NUM_WARPS * 64))],
            threads_per_warp=[64],
            warps_per_cta=[NUM_WARPS],
            order=[0],
        )

        offs = gl.arange(0, BLOCK_E, layout=layout)
        mask = offs < E

        row_ptr = logits_ptr + pid.to(gl.int64) * logits_stride_m
        logits = gl.load(row_ptr + offs, mask=mask, other=float("-inf")).to(gl.float32)

        # Numerically stable softmax in fp32.
        row_max = gl.max(logits, axis=0)
        shifted = logits - row_max
        exp_logits = gl.exp(shifted)
        # exp(-inf) = 0, so masked lanes contribute nothing to the sum.
        denom = gl.sum(exp_logits, axis=0)
        scores = exp_logits / denom    # softmax distribution, masked lanes -> 0

        # Iterative top-k. Each iteration:
        #   1. max-reduce to find the largest score.
        #   2. min-reduce over indices marked with the max value to break ties
        #      with the lowest expert id (matching torch.topk's convention).
        #   3. record (val, idx), then set that slot to -inf so it is not
        #      selected again.
        weight_sum = 0.0
        out_w_row = out_weights_ptr + pid.to(gl.int64) * weights_stride_m
        out_i_row = out_ids_ptr + pid.to(gl.int64) * ids_stride_m

        for k in gl.static_range(TOP_K):
            top_val = gl.max(scores, axis=0)
            # mark indices that hold the max with their actual id, others with E.
            id_marker = gl.where(scores == top_val, offs, E)
            top_idx = gl.min(id_marker, axis=0)

            gl.store(out_w_row + k, top_val)
            gl.store(out_i_row + k, top_idx)
            weight_sum = weight_sum + top_val

            # Mask out the selected slot.
            scores = gl.where(offs == top_idx, float("-inf"), scores)

        if RENORMALIZE:
            inv = 1.0 / weight_sum
            for k in gl.static_range(TOP_K):
                w = gl.load(out_w_row + k)
                gl.store(out_w_row + k, w * inv)


def invoke_gluon_topk_softmax(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    renormalize: bool = True,
    num_warps: int = 1,
) -> None:
    """Compute top-k softmax of router logits.

    Args:
        logits: ``(M, E)`` fp16/bf16/fp32 router logits.
        topk_weights: ``(M, top_k)`` fp32 output buffer.
        topk_ids: ``(M, top_k)`` int32 output buffer.
        renormalize: if True, divide the top-k weights by their sum.
    """
    assert _GLUON_AVAILABLE, "Gluon is not available in this Triton build"
    assert logits.dim() == 2
    M, E = logits.shape
    assert topk_weights.shape[0] == M and topk_ids.shape[0] == M
    assert topk_weights.shape[1] == topk_ids.shape[1]
    top_k = topk_weights.shape[1]
    assert topk_weights.dtype == torch.float32
    assert topk_ids.dtype == torch.int32

    BLOCK_E = triton.next_power_of_2(E)
    if BLOCK_E < 64:
        BLOCK_E = 64

    gluon_topk_softmax_kernel[(M,)](
        logits,
        topk_weights,
        topk_ids,
        logits.stride(0),
        topk_weights.stride(0),
        topk_ids.stride(0),
        E,
        TOP_K=top_k,
        BLOCK_E=BLOCK_E,
        RENORMALIZE=renormalize,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
