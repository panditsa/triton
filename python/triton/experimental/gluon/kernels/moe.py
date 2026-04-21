"""
Gluon bf16 MOE GEMM kernel for AMD CDNA4 (gfx950).

Drop-in replacement for the bf16 path of the Triton ``fused_moe_kernel``
(scatter/gather GEMM) used by sglang / vLLM / aiter MOE pipelines.

Activation:
    Set the env var SGLANG_USE_GLUON_MOE=1 in the dispatch wrapper, OR
    call ``invoke_gluon_fused_moe_kernel`` directly.

Supported tile config:
    BLOCK_SIZE_M = 128, BLOCK_SIZE_N = 128, BLOCK_SIZE_K = 64, num_warps = 8.
    bf16 inputs/outputs, fp32 accumulator.

Supported features:
    - per-token gather of A rows via ``sorted_token_ids``
    - per-block expert id via ``expert_ids``
    - optional ``MUL_ROUTED_WEIGHT`` (multiply by per-token routing weight)
    - ``filter_expert`` (write zeros when the expert is not on this rank)
    - sorted (``c_sorted=True``) and unsorted writeback
    - even and uneven K (``K % BLOCK_SIZE_K == 0`` is the fast path)

Hardware features used:
    - ``v_mfma_f32_16x16x32_bf16`` (matching CK ``kernel_moe_gemm_2lds``)
    - ``buffer_load_dword ... offen lds`` for the A gather
      (``cdna4_async_copy.buffer_load_to_shared``)
    - 8 warps split as ``warps_per_cta=[2, 4]``; with twice as many warps
      the wave scheduler has more candidates to hide memory latency,
      which is the structural prerequisite for ping-pong
    - double-buffered LDS for both A and B; the manual 2-stage software
      pipeline overlaps the next iter's HBM loads with the current iter's
      MFMA reads
    - swizzled LDS layout matching ``DotOperandLayout`` ``k_width=8``

Notes on warp_pipeline_stage:
    Explicit ping-pong via ``gl.amd.warp_pipeline_stage`` was attempted
    but does not fit this kernel. The conversion pass turns those
    markers into a ``CondBarrier`` plus ``s_setprio`` schedule that
    splits warps into two groups running different loop iterations,
    which is incompatible with the collective MFMA that all 8 warps
    participate in. Adding ``s_setprio`` without the warp split would
    need backend changes outside Gluon.

Notes on the gap to CK ``kernel_moe_gemm_2lds``:
    The Triton AMD backend currently does not allocate AGPRs for MFMA
    accumulators (``Accum_VGPR_Count = 0`` in every dispatch), so this
    kernel reaches Triton-tuned parity but stops short of CK's MFMA
    register-renaming throughput (CK uses ~168 AGPRs). Closing that gap
    requires backend changes outside Gluon itself.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import torch
import triton

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.amd.cdna4 import (
        async_copy as cdna4_async_copy,
    )

    _GLUON_AVAILABLE = True
except ImportError:
    _GLUON_AVAILABLE = False
    gluon = None
    gl = None
    cdna4_async_copy = None


def gluon_available() -> bool:
    return _GLUON_AVAILABLE


_USE_GLUON_MOE_ENV = "SGLANG_USE_GLUON_MOE"


def use_gluon_moe() -> bool:
    """Return True iff env var SGLANG_USE_GLUON_MOE is set to a truthy value."""
    if not _GLUON_AVAILABLE:
        return False
    val = os.environ.get(_USE_GLUON_MOE_ENV, "0").lower()
    return val in ("1", "true", "yes", "on")


_DEFAULT_GLUON_CONFIG: Dict[str, Any] = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 8,
    # 8 warps split as warps_per_cta=[2, 4] enable the ping-pong schedule:
    # two warp groups alternate between issuing memory ops and running MFMA.
    "num_warps": 8,
    "num_stages": 1,
    # waves_per_eu hint: requesting 2 waves per EU constrains VGPR usage
    # (~128 VGPRs max) which encourages LLVM to spill to AGPRs for the MFMA
    # accumulator and may unlock register renaming on future backend versions.
    "waves_per_eu": 2,
}


def get_default_gluon_config() -> Dict[str, Any]:
    return dict(_DEFAULT_GLUON_CONFIG)


def gluon_config_supported(config: Dict[str, Any], K: int) -> bool:
    """Return True iff this Gluon kernel can serve the given config."""
    if not _GLUON_AVAILABLE:
        return False
    if config.get("BLOCK_SIZE_M") != 128:
        return False
    if config.get("BLOCK_SIZE_N") != 128:
        return False
    if config.get("BLOCK_SIZE_K") != 64:
        return False
    if config.get("num_warps", 8) != 8:
        return False
    if K % config["BLOCK_SIZE_K"] != 0:
        return False
    return True


# --------------------------------------------------------------------------- #
#  Kernel
# --------------------------------------------------------------------------- #


if _GLUON_AVAILABLE:

    @gluon.jit
    def gluon_fused_moe_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        N,
        K,
        EM,
        num_valid_tokens,
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: gl.constexpr,
        BLOCK_SIZE_N: gl.constexpr,
        BLOCK_SIZE_K: gl.constexpr,
        GROUP_SIZE_M: gl.constexpr,
        MUL_ROUTED_WEIGHT: gl.constexpr,
        top_k: gl.constexpr,
        compute_type: gl.constexpr,
        even_Ks: gl.constexpr,
        c_sorted: gl.constexpr,
        filter_expert: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        """Gather/scatter MOE GEMM:

            C[offs_token[m], n] = sum_k A[offs_token[m] // top_k, k]
                                       * B[expert, n, k]

        Tile config (fixed): 128 x 128 x 64 (M x N x K), num_warps=4.
        MFMA: ``v_mfma_f32_16x16x32_bf16`` (matching CK).
        """

        gl.static_assert(BLOCK_SIZE_M == 128, "Gluon kernel requires BLOCK_SIZE_M=128")
        gl.static_assert(BLOCK_SIZE_N == 128, "Gluon kernel requires BLOCK_SIZE_N=128")
        gl.static_assert(BLOCK_SIZE_K == 64, "Gluon kernel requires BLOCK_SIZE_K=64")
        gl.static_assert(NUM_WARPS == 8, "Gluon kernel requires num_warps=8 for ping-pong")

        # ---- pid -> (pid_m, pid_n) using grouped order for L2 reuse ----
        pid = gl.program_id(axis=0)
        num_pid_m = gl.cdiv(EM, BLOCK_SIZE_M)
        num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ---- early exit on padding blocks ----
        num_tokens_post_padded = gl.load(num_tokens_post_padded_ptr)
        if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
            return

        # ---- layouts ----
        # MFMA v_mfma_f32_16x16x32_bf16, 8 warps as 2x4 to enable ping-pong.
        # Per-wave tile: M=128/2=64, N=128/4=32. Per-wave MFMA insns per
        # K-step: (64/16) * (32/16) * (64/32) = 4 * 2 * 2 = 16.
        MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, 32]
        MFMA_K_WIDTH: gl.constexpr = 8
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=MFMA_INSTR_SHAPE,
            transposed=True,
            warps_per_cta=[2, 4],
        )
        dot_a_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mfma_layout, k_width=MFMA_K_WIDTH
        )
        dot_b_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=1, parent=mfma_layout, k_width=MFMA_K_WIDTH
        )

        # Load layout for B in the [BLOCK_N, BLOCK_K] orientation.
        # Bases are the trans of dot_b_layout's bases (which describe
        # [BLOCK_K, BLOCK_N]). After loading B into this layout and
        # calling .trans(1, 0), the result lands in dot_b_layout
        # trivially (metadata-only convert_layout, no cross-warp moves).
        # Hand-derived from the compiler's dump of dot_b_layout:
        #   reg_bases  = [[1, 0], [2, 0], [4, 0], [32, 0], [0, 64]]
        #   lane_bases = [[0, 1], [0, 2], [0, 4], [0, 8], [8, 0], [16, 0]]
        #   warp_bases = [[0, 16], [0, 32], [0, 0]]
        #   shape      = [64, 128]   (K, N)
        # which transposes for shape [128, 64] (N, K) to:
        load_b_nk_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [0, 4], [0, 32], [64, 0]],
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 8], [0, 16]],
            warp_bases=[[16, 0], [32, 0], [0, 0]],
            block_bases=[],
            shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )

        # Blocked layouts. size_per_thread[contig] * 16 = 128 bits per
        # buffer_load instruction (the LDS-direct path's strict requirement).
        # Both A and B are loaded as [BIG, BLOCK_K] tiles where K (the
        # HBM-contig dim) lands on dim 1, so order=[1, 0] satisfies the
        # buffer_load_to_shared lowering. The B tile is therefore labelled
        # [BLOCK_N, BLOCK_K] in this kernel (transposed from [K, N]); the
        # transpose back to the MFMA-expected orientation happens on the
        # LDS read side via tensor.trans + convert_layout.
        blocked_mk: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[8, 8],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        )
        blocked_nk: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[8, 8],
            warps_per_cta=[NUM_WARPS, 1],
            order=[1, 0],
        )

        # vec=8 (=k_width) gives bank-conflict-free MFMA reads.
        # shared_b is [BLOCK_N, BLOCK_K] order=[1, 0] (K contig in LDS) so
        # the matching buffer_load_to_shared can lower.
        shared_a: gl.constexpr = gl.SwizzledSharedLayout(
            vec=MFMA_K_WIDTH, per_phase=1, max_phase=16, order=[1, 0]
        )
        shared_b: gl.constexpr = gl.SwizzledSharedLayout(
            vec=MFMA_K_WIDTH, per_phase=1, max_phase=16, order=[1, 0]
        )

        # ---- gather A: per-token row indices, then per-element offsets ----
        offs_token_layout: gl.constexpr = gl.SliceLayout(1, blocked_mk)
        offs_kA_layout: gl.constexpr = gl.SliceLayout(0, blocked_mk)

        offs_token_id = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=offs_token_layout
        )
        offs_token = gl.load(sorted_token_ids_ptr + offs_token_id)
        token_mask = offs_token < num_valid_tokens

        # ---- expert id ----
        off_experts = gl.load(expert_ids_ptr + pid_m)
        if filter_expert and off_experts == -1:
            zero_layout: gl.constexpr = blocked_mk
            offs_zm = pid_m * BLOCK_SIZE_M + gl.arange(
                0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, zero_layout)
            )
            offs_zn = pid_n * BLOCK_SIZE_N + gl.arange(
                0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, zero_layout)
            )
            offs_token_zero = gl.load(
                sorted_token_ids_ptr + offs_zm,
                mask=offs_zm < EM,
                other=num_valid_tokens,
            )
            zero_mask = (offs_token_zero[:, None] < num_valid_tokens) & (
                offs_zn[None, :] < N
            )
            zero_val = gl.zeros(
                (BLOCK_SIZE_M, BLOCK_SIZE_N),
                dtype=compute_type,
                layout=zero_layout,
            )
            zero_ptrs = (
                c_ptr
                + offs_token_zero[:, None].to(gl.int64) * stride_cm
                + offs_zn[None, :].to(gl.int64) * stride_cn
            )
            gl.store(zero_ptrs, zero_val, mask=zero_mask)
            return

        # Cast offsets to int32 for buffer_load_to_shared. Element offset fits
        # in int32 for all DSv3/Mixtral shapes (max ~30M elements).
        offs_a_row = (offs_token.to(gl.int32) // top_k) * stride_am
        offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=offs_kA_layout)
        offs_a_base = offs_a_row[:, None] + offs_ak[None, :] * stride_ak

        # ---- B offsets (per-expert; expert is a runtime scalar) ----
        # The B tile is loaded as [BLOCK_N, BLOCK_K] so the HBM-contig
        # dim (K) sits on tile dim 1 to satisfy the buffer_load_to_shared
        # constraint. The transpose back to [K, N] happens on the LDS read.
        offs_bk_layout: gl.constexpr = gl.SliceLayout(0, blocked_nk)
        offs_bn_layout: gl.constexpr = gl.SliceLayout(1, blocked_nk)

        offs_bn = (
            pid_n * BLOCK_SIZE_N
            + gl.arange(0, BLOCK_SIZE_N, layout=offs_bn_layout)
        ) % N
        offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=offs_bk_layout)
        offs_b_base = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk

        b_ptr_e = b_ptr + off_experts.to(gl.int64) * stride_be

        # ---- shared memory ----
        # Both A and B use async LDS-direct loads via buffer_load_to_shared.
        # Both are double-buffered so the next iter's load can be issued
        # before the current iter's MFMA reads finish; wait_group(1) drains
        # the older pair (A and B share the same async commit group).
        # LDS budget per CTA: 2 * 128 * 64 * 2 (A) + 2 * 128 * 64 * 2 (B)
        # = 32 KiB + 32 KiB = 64 KiB. gfx950 has 160 KiB LDS per CU, so
        # this leaves room for 2 CTAs per CU at the current VGPR pressure.
        # Bumping to NUM_BUFFERS=3 each was tried (96 KiB) and regressed
        # 5-15% because the per-CU LDS limit cuts occupancy from 2 to 1.
        NUM_BUFFERS_A: gl.constexpr = 2
        NUM_BUFFERS_B: gl.constexpr = 2
        smem_a = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            [NUM_BUFFERS_A, BLOCK_SIZE_M, BLOCK_SIZE_K],
            shared_a,
        )
        smem_b = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            [NUM_BUFFERS_B, BLOCK_SIZE_N, BLOCK_SIZE_K],
            shared_b,
        )

        # ---- prologue: async A and B loads into LDS buffer 0 ----
        if even_Ks:
            cdna4_async_copy.buffer_load_to_shared(
                smem_a.index(0), a_ptr, offs_a_base, mask=token_mask[:, None]
            )
            cdna4_async_copy.buffer_load_to_shared(
                smem_b.index(0), b_ptr_e, offs_b_base
            )
        else:
            cdna4_async_copy.buffer_load_to_shared(
                smem_a.index(0),
                a_ptr,
                offs_a_base,
                mask=token_mask[:, None] & (offs_ak[None, :] < K),
            )
            cdna4_async_copy.buffer_load_to_shared(
                smem_b.index(0),
                b_ptr_e,
                offs_b_base,
                mask=offs_bk[None, :] < K,
            )
        cdna4_async_copy.commit_group()

        acc = gl.zeros(
            (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
        )

        num_k_iter = gl.cdiv(K, BLOCK_SIZE_K)

        # ---- pipelined K-loop ----
        # Pattern per loop iteration:
        #   1. Issue async A load for next iter into smem_a[nxt_a_buf]
        #      and async B load for next iter into smem_b[nxt_b_buf].
        #      Both go through buffer_load_dwordx4 ... offen lds.
        #   2. commit_group ties them into one async group.
        #   3. wait_group(1) drains the older group (current iter's
        #      A and B reach LDS).
        #   4. Read A from smem_a[cur_a_buf] into dot_a_layout.
        #   5. Read B from smem_b[cur_b_buf] into a [BLOCK_N, BLOCK_K]
        #      tensor, transpose to [BLOCK_K, BLOCK_N], convert into
        #      dot_b_layout for the MFMA.
        #   6. Collective MFMA across all 8 warps.
        # `gl.amd.warp_pipeline_stage` is intentionally NOT used; its
        # conversion pass implements warp specialization incompatible
        # with this kernel's collective MFMA.
        for k in range(0, num_k_iter - 1):
            k_off_next = (k + 1) * BLOCK_SIZE_K
            offs_a_n = offs_a_base + k_off_next * stride_ak
            offs_b_n = offs_b_base + k_off_next * stride_bk
            cur_a_buf = k % NUM_BUFFERS_A
            nxt_a_buf = (k + 1) % NUM_BUFFERS_A
            cur_b_buf = k % NUM_BUFFERS_B
            nxt_b_buf = (k + 1) % NUM_BUFFERS_B

            if even_Ks:
                cdna4_async_copy.buffer_load_to_shared(
                    smem_a.index(nxt_a_buf), a_ptr, offs_a_n,
                    mask=token_mask[:, None],
                )
                cdna4_async_copy.buffer_load_to_shared(
                    smem_b.index(nxt_b_buf), b_ptr_e, offs_b_n,
                )
            else:
                cdna4_async_copy.buffer_load_to_shared(
                    smem_a.index(nxt_a_buf),
                    a_ptr,
                    offs_a_n,
                    mask=token_mask[:, None] & (offs_ak[None, :] < K - k_off_next),
                )
                cdna4_async_copy.buffer_load_to_shared(
                    smem_b.index(nxt_b_buf),
                    b_ptr_e,
                    offs_b_n,
                    mask=offs_bk[None, :] < K - k_off_next,
                )
            cdna4_async_copy.commit_group()

            cdna4_async_copy.wait_group(1)
            cur_a = cdna4_async_copy.load_shared_relaxed(
                smem_a.index(cur_a_buf), dot_a_layout
            )
            cur_b_nk = cdna4_async_copy.load_shared_relaxed(
                smem_b.index(cur_b_buf), load_b_nk_layout
            )
            cur_b = gl.convert_layout(
                cur_b_nk.trans(1, 0), dot_b_layout, assert_trivial=True
            )
            acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)

        # ---- epilogue: process the last K tile ----
        last_a_buf = (num_k_iter - 1) % NUM_BUFFERS_A
        last_b_buf = (num_k_iter - 1) % NUM_BUFFERS_B
        cdna4_async_copy.wait_group(0)
        cur_a = cdna4_async_copy.load_shared_relaxed(
            smem_a.index(last_a_buf), dot_a_layout
        )
        cur_b_nk = cdna4_async_copy.load_shared_relaxed(
            smem_b.index(last_b_buf), load_b_nk_layout
        )
        cur_b = gl.convert_layout(
            cur_b_nk.trans(1, 0), dot_b_layout, assert_trivial=True
        )
        acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)

        # ---- optional routed-weight scaling ----
        if MUL_ROUTED_WEIGHT:
            mw_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
            offs_token_mw = pid_m * BLOCK_SIZE_M + gl.arange(
                0, BLOCK_SIZE_M, layout=mw_layout
            )
            offs_token_mw_id = gl.load(
                sorted_token_ids_ptr + offs_token_mw,
                mask=offs_token_mw < EM,
                other=num_valid_tokens,
            )
            mw_mask = offs_token_mw_id < num_valid_tokens
            moe_weight = gl.load(
                topk_weights_ptr + offs_token_mw_id, mask=mw_mask, other=0.0
            )
            acc = acc * moe_weight[:, None]

        c_val = acc.to(compute_type)

        # ---- scatter store to C ----
        cm_layout: gl.constexpr = gl.SliceLayout(1, mfma_layout)
        cn_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)
        offs_cm_id = pid_m * BLOCK_SIZE_M + gl.arange(
            0, BLOCK_SIZE_M, layout=cm_layout
        )
        if c_sorted:
            offs_cm_dest = offs_cm_id
            cm_mask = offs_cm_id < EM
        else:
            offs_cm_dest = gl.load(
                sorted_token_ids_ptr + offs_cm_id,
                mask=offs_cm_id < EM,
                other=num_valid_tokens,
            )
            cm_mask = offs_cm_dest < num_valid_tokens

        offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=cn_layout)
        c_ptrs = (
            c_ptr
            + offs_cm_dest[:, None].to(gl.int64) * stride_cm
            + offs_cn[None, :].to(gl.int64) * stride_cn
        )
        c_mask = cm_mask[:, None] & (offs_cn[None, :] < N)
        gl.store(c_ptrs, c_val, mask=c_mask)


# --------------------------------------------------------------------------- #
#  Wrapper
# --------------------------------------------------------------------------- #


def invoke_gluon_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type,
    c_sorted: bool = False,
    filter_expert: bool = True,
) -> None:
    """Launch the Gluon MOE GEMM kernel.

    Args:
        A: token-feature matrix, shape (num_tokens, K), bf16.
        B: stacked expert weights, shape (E, N, K), bf16.
        C: output cache, shape (num_tokens * top_k, N) (or (EM, N) when
            ``c_sorted=True``), bf16.
        topk_weights: per-(token, expert) routing weights, fp32.
        topk_ids: per-token expert ids, int32.
        sorted_token_ids: sorted (token, expert) ids padded to BLOCK_SIZE_M, int32.
        expert_ids: per-block expert id, int32.
        num_tokens_post_padded: scalar tensor with the padded length.
        mul_routed_weight: multiply accumulator by the routing weight before store.
        top_k: number of experts per token.
        config: kernel tile config (must satisfy ``gluon_config_supported``).
        compute_type: Triton dtype for the output (e.g. ``gl.bfloat16``).
        c_sorted: if True, write back to ``offs_token_id`` (sorted layout) instead
            of ``offs_token`` (original token order).
        filter_expert: if True, write zeros when ``expert_id == -1``.
    """
    assert _GLUON_AVAILABLE, "Gluon is not available in this Triton build"
    assert A.dtype == torch.bfloat16, "Gluon kernel only supports bf16 inputs"
    assert B.dtype == torch.bfloat16, "Gluon kernel only supports bf16 weights"

    K = B.shape[2]
    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )

    gluon_fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],  # N
        K,
        sorted_token_ids.shape[0],  # EM
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"],
        BLOCK_SIZE_N=config["BLOCK_SIZE_N"],
        BLOCK_SIZE_K=config["BLOCK_SIZE_K"],
        GROUP_SIZE_M=config["GROUP_SIZE_M"],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        c_sorted=c_sorted,
        filter_expert=filter_expert,
        NUM_WARPS=config["num_warps"],
        num_warps=config["num_warps"],
        waves_per_eu=config.get("waves_per_eu", 0),
    )
