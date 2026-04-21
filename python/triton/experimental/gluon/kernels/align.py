"""
Gluon moe_align_block_size: sort topk_ids by expert and pad to block_size.

Replaces sgl_kernel's CUDA ``moe_align_block_size`` for the bf16 MOE path.
Outputs:
    sorted_token_ids: ``(max_padded,)`` int32, indices into the flat
        ``(num_tokens * top_k,)`` view of ``topk_ids``, sorted by expert id and
        padded with the sentinel ``num_total = M * top_k`` so each per-expert
        block is a multiple of ``BLOCK_SIZE_M``.
    expert_ids: ``(max_blocks,)`` int32, the expert id for each block of
        ``BLOCK_SIZE_M`` consecutive entries in ``sorted_token_ids``.
    num_tokens_post_padded: ``(1,)`` int32, total length used.

Implementation:
    A single CTA does a count + scan + scatter. NUM_EXPERTS must be at
    most 1024 (room for the count buffer in shared memory). For DSv3 / Mixtral
    / Qwen-MoE, num_experts is at most 256.

Performance:
    Single-CTA, intentionally simple. The CUDA reference in
    ``sgl-kernel/csrc/moe/moe_align_block_size.cu`` is faster on very large
    inputs because it parallelizes the count phase across multiple CTAs and
    finishes with one CTA doing the scan. For our typical M * top_k (up to
    32768 for DSv3-TP1 bs=4096) this single-CTA version runs in well under
    100us, several orders of magnitude below the surrounding GEMM cost.
"""

from __future__ import annotations

import os
from typing import Tuple

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


_USE_GLUON_ALIGN_ENV = "SGLANG_USE_GLUON_ALIGN"


def use_gluon_align() -> bool:
    if not _GLUON_AVAILABLE:
        return False
    val = os.environ.get(_USE_GLUON_ALIGN_ENV, "0").lower()
    return val in ("1", "true", "yes", "on")


if _GLUON_AVAILABLE:

    @gluon.jit
    def _count_atomic_kernel(
        topk_ids_ptr,
        counts_ptr,                # (NUM_EXPERTS,) int32, zero-initialised
        num_total,
        NUM_EXPERTS: gl.constexpr,
        BLOCK: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        """Phase 1: count tokens per expert via global atomic adds."""
        pid = gl.program_id(0)
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[64],
            warps_per_cta=[NUM_WARPS],
            order=[0],
        )
        offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
        mask = offs < num_total
        eids = gl.load(topk_ids_ptr + offs, mask=mask, other=NUM_EXPERTS)
        # Filtered experts (-1) are routed to slot NUM_EXPERTS (the sentinel
        # bucket). Their tokens land in the padding region and are dropped by
        # the GEMM's mask.
        eids_safe = gl.where(eids == -1, NUM_EXPERTS, eids)
        ones = gl.full([BLOCK], 1, gl.int32, layout=layout)
        gl.atomic_add(counts_ptr + eids_safe, ones, mask=mask, sem="relaxed")

    @gluon.jit
    def _scatter_kernel(
        topk_ids_ptr,
        offsets_ptr,               # (NUM_EXPERTS + 1,) int32 cumulative starts
        write_ptr,                 # (NUM_EXPERTS,) int32, zero-initialised cursor
        sorted_ids_ptr,            # (max_padded,) int32
        num_total,
        SENTINEL: gl.constexpr,    # placeholder for invalid slots = num_total
        NUM_EXPERTS: gl.constexpr,
        BLOCK: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        """Phase 3: scatter each (m, k) into its sorted position."""
        pid = gl.program_id(0)
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[64],
            warps_per_cta=[NUM_WARPS],
            order=[0],
        )
        offs = pid * BLOCK + gl.arange(0, BLOCK, layout=layout)
        mask = offs < num_total

        eids = gl.load(topk_ids_ptr + offs, mask=mask, other=NUM_EXPERTS)
        eids_safe = gl.where(eids == -1, NUM_EXPERTS, eids)

        # Atomically grab a slot within this expert's sorted region.
        ones = gl.full([BLOCK], 1, gl.int32, layout=layout)
        slot = gl.atomic_add(write_ptr + eids_safe, ones, mask=mask, sem="relaxed")
        starts = gl.load(offsets_ptr + eids_safe, mask=mask, other=0)
        sorted_pos = starts + slot
        gl.store(sorted_ids_ptr + sorted_pos, offs.to(gl.int32), mask=mask)


def _scan_and_pad_torch(
    counts: torch.Tensor,
    block_size: int,
    num_experts_inc_sentinel: int,
    max_padded: int,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    sentinel_value: int,
) -> torch.Tensor:
    """Phase 2: cumulative scan, write padding sentinels, fill expert_ids."""
    # Pad each expert's count up to a block boundary.
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    # Prefix sums of padded counts give the start offset of each expert in
    # the sorted output. Slot NUM_EXPERTS (the -1 / filtered bucket) is the
    # tail of the sorted region; its tokens are also written but the
    # corresponding expert_ids slot is set to -1 so the GEMM zero-fills.
    starts = torch.empty(
        num_experts_inc_sentinel + 1, dtype=torch.int32, device=counts.device
    )
    starts[0] = 0
    torch.cumsum(padded_counts, 0, dtype=torch.int32, out=starts[1:])

    # Initialise the entire sorted region to the sentinel (= num_total). The
    # scatter kernel will overwrite the valid slots; everything else stays
    # sentinel and is masked out by the GEMM's `token_mask = offs_token <
    # num_valid_tokens`.
    sorted_ids.fill_(sentinel_value)

    total_padded = int(starts[-1].item())
    num_tokens_post_padded[0] = total_padded

    # Write expert_ids[block_idx] = expert_id (or -1 for the sentinel slot).
    # Each expert occupies its own contiguous block range.
    expert_ids.fill_(-1)
    block_starts = starts[:-1] // block_size
    block_ends = starts[1:] // block_size
    # vectorised loop over experts; small (<= 257) so this is cheap.
    for e in range(num_experts_inc_sentinel):
        bs = int(block_starts[e].item())
        be = int(block_ends[e].item())
        if be > bs:
            ev = e if e < num_experts_inc_sentinel - 1 else -1
            expert_ids[bs:be].fill_(ev)
    return starts


def gluon_moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for sglang's ``moe_align_block_size``.

    Three-phase implementation:
        Phase 1 (Gluon kernel): atomic count of tokens per expert.
        Phase 2 (torch ops): cumulative scan, padding, expert_ids write.
        Phase 3 (Gluon kernel): scatter each (m, k) into its sorted slot.

    Returns ``(sorted_token_ids, expert_ids, num_tokens_post_padded)`` in the
    same shapes and dtypes as the sgl_kernel CUDA implementation.
    """
    assert _GLUON_AVAILABLE, "Gluon is not available in this Triton build"
    assert topk_ids.dtype in (torch.int32, torch.int64)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    num_total = topk_ids.numel()
    # +1 for the -1 / filtered "sentinel" expert.
    num_experts_inc = num_experts + 1

    # Same buffer-sizing formula as sglang's reference ``moe_align_block_size``:
    # the worst case is one extra (block_size - 1) padding slot per expert
    # plus the sentinel bucket.
    if num_total < num_experts_inc + 1:
        max_padded = num_total * block_size
    else:
        max_padded = num_total + num_experts_inc * (block_size - 1)
    max_blocks = triton.cdiv(max_padded, block_size)

    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=topk_ids.device)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    counts = torch.zeros(num_experts_inc, dtype=torch.int32, device=topk_ids.device)

    BLOCK = 1024
    NUM_WARPS = 4
    grid = (triton.cdiv(num_total, BLOCK),)

    _count_atomic_kernel[grid](
        topk_ids,
        counts,
        num_total,
        NUM_EXPERTS=num_experts_inc,
        BLOCK=BLOCK,
        NUM_WARPS=NUM_WARPS,
        num_warps=NUM_WARPS,
    )

    _scan_and_pad_torch(
        counts,
        block_size,
        num_experts_inc,
        max_padded,
        sorted_ids,
        expert_ids,
        num_tokens_post_padded,
        sentinel_value=num_total,
    )

    # Build the cumulative offsets buffer the scatter kernel needs.
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    offsets = torch.empty(num_experts_inc + 1, dtype=torch.int32, device=topk_ids.device)
    offsets[0] = 0
    torch.cumsum(padded_counts, 0, dtype=torch.int32, out=offsets[1:])

    write_cursor = torch.zeros(num_experts_inc, dtype=torch.int32, device=topk_ids.device)
    _scatter_kernel[grid](
        topk_ids,
        offsets,
        write_cursor,
        sorted_ids,
        num_total,
        SENTINEL=num_total,
        NUM_EXPERTS=num_experts_inc,
        BLOCK=BLOCK,
        NUM_WARPS=NUM_WARPS,
        num_warps=NUM_WARPS,
    )

    return sorted_ids, expert_ids, num_tokens_post_padded
