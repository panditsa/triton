"""
Unit tests for the MOE Gluon kernels in
``triton.experimental.gluon.kernels``. One test per sublock plus an
integration test that chains all five through a torch reference.

The tests are gfx950-only (CDNA4 MFMA, CDNA4 async LDS-direct loads).

Run from the Triton checkout:

    pytest python/test/gluon/test_kernels_moe.py -v
    pytest python/test/gluon/test_kernels_moe.py::test_topk_softmax -v
    pytest python/test/gluon/test_kernels_moe.py -k silu_and_mul -v
"""

import pytest
import torch
import torch.nn.functional as F

from triton._internal_testing import is_hip_cdna4
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.kernels import (
    gluon_moe_align_block_size,
    invoke_gluon_act_and_mul,
    invoke_gluon_fused_moe_kernel,
    invoke_gluon_moe_sum_reduce,
    invoke_gluon_topk_softmax,
)


pytestmark = pytest.mark.skipif(
    not is_hip_cdna4(), reason="MOE Gluon kernels require CDNA4 (gfx950)"
)


# --------------------------------------------------------------------------- #
#  topk_softmax
# --------------------------------------------------------------------------- #


def _sort_pairs_by_id(weights, ids):
    """topk(softmax) order is implementation-defined when scores tie. Sort by
    id within each row before comparing."""
    order = ids.argsort(dim=-1)
    return weights.gather(-1, order), ids.gather(-1, order)


@pytest.mark.parametrize("E,top_k", [(8, 2), (60, 4), (256, 8)])
@pytest.mark.parametrize("M", [64, 1024])
@pytest.mark.parametrize("renormalize", [True, False])
def test_topk_softmax(E, top_k, M, renormalize):
    torch.manual_seed(0)
    logits = torch.randn(M, E, device="cuda", dtype=torch.float32)

    out_w = torch.empty(M, top_k, device="cuda", dtype=torch.float32)
    out_i = torch.empty(M, top_k, device="cuda", dtype=torch.int32)
    invoke_gluon_topk_softmax(logits, out_w, out_i, renormalize=renormalize)

    sm = torch.softmax(logits.double(), dim=-1)
    ref_w_full, ref_i = torch.topk(sm, top_k)
    if renormalize:
        ref_w_full = ref_w_full / ref_w_full.sum(dim=-1, keepdim=True)
    ref_w = ref_w_full.to(torch.float32)
    ref_i = ref_i.to(torch.int32)

    out_w, out_i = _sort_pairs_by_id(out_w, out_i)
    ref_w, ref_i = _sort_pairs_by_id(ref_w, ref_i)

    torch.testing.assert_close(out_w, ref_w, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(out_i, ref_i)


# --------------------------------------------------------------------------- #
#  moe_align_block_size
# --------------------------------------------------------------------------- #


def _per_expert_counts(sorted_ids, expert_ids, npp_t, num_total, block_size):
    npp = int(npp_t.item())
    n_blocks = npp // block_size
    counts = {}
    for b in range(n_blocks):
        e = int(expert_ids[b].item())
        if e == -1:
            continue
        blk = sorted_ids[b * block_size:(b + 1) * block_size]
        counts[e] = counts.get(e, 0) + int((blk < num_total).sum().item())
    return counts


@pytest.mark.parametrize(
    "E,top_k,M,block_size",
    [
        (8, 2, 64, 64),
        (8, 2, 256, 128),
        (60, 4, 256, 128),
        (256, 8, 1024, 128),
    ],
)
def test_moe_align_block_size(E, top_k, M, block_size):
    torch.manual_seed(0)
    sc = torch.softmax(torch.randn(M, E, device="cuda"), dim=-1)
    _, ti = torch.topk(sc, top_k)
    ti = ti.to(torch.int32)

    sorted_ids, expert_ids, npp = gluon_moe_align_block_size(ti, block_size, E)

    # Per-expert valid-token counts must match a torch.bincount reference.
    got = _per_expert_counts(sorted_ids, expert_ids, npp, M * top_k, block_size)
    ref = {
        e: int(c)
        for e, c in enumerate(torch.bincount(ti.flatten().long(), minlength=E).tolist())
        if c > 0
    }
    assert got == ref, f"per-expert count mismatch:\n  got: {got}\n  ref: {ref}"

    # Each expert must be padded to a multiple of block_size.
    npp_v = int(npp.item())
    for blk in range(npp_v // block_size):
        e = int(expert_ids[blk].item())
        if e == -1:
            continue
        # Block contents: each id is either a valid (m, k) pair < num_total
        # or the sentinel value M * top_k.
        contents = sorted_ids[blk * block_size:(blk + 1) * block_size]
        # No ids out of range.
        assert (contents <= M * top_k).all().item()
        # Valid ids in this block must all route to expert `e`.
        valid = contents[contents < M * top_k]
        if valid.numel() > 0:
            assigned = ti.flatten()[valid.long()]
            assert (assigned == e).all().item(), \
                f"block {blk} (expert {e}) contains ids assigned to a different expert"


# --------------------------------------------------------------------------- #
#  silu_and_mul / gelu_and_mul
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("M_total", [16, 64, 512])
@pytest.mark.parametrize("I", [256, 1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_silu_and_mul(M_total, I, dtype):
    torch.manual_seed(0)
    gateup = torch.randn(M_total, 2 * I, device="cuda", dtype=dtype) * 0.2

    out = torch.empty(M_total, I, device="cuda", dtype=dtype)
    invoke_gluon_act_and_mul(gateup, out, activation="silu")

    gate = gateup[:, :I].float()
    up = gateup[:, I:].float()
    ref = (F.silu(gate) * up).to(dtype)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


@pytest.mark.parametrize("M_total,I", [(32, 512), (128, 2048)])
def test_gelu_and_mul(M_total, I):
    torch.manual_seed(0)
    gateup = torch.randn(M_total, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.2

    out = torch.empty(M_total, I, device="cuda", dtype=torch.bfloat16)
    invoke_gluon_act_and_mul(gateup, out, activation="gelu")

    gate = gateup[:, :I].float()
    up = gateup[:, I:].float()
    ref = (F.gelu(gate, approximate="tanh") * up).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=1e-2)


def test_silu_and_mul_filter_expert():
    """Rows whose expert id is -1 are skipped; the corresponding output rows
    must not be touched."""
    torch.manual_seed(0)
    M_total, I, expert_step = 8, 256, 1
    gateup = torch.randn(M_total, 2 * I, device="cuda", dtype=torch.bfloat16) * 0.2
    expert_ids = torch.tensor([0, -1, 1, -1, 2, 3, -1, 4],
                              device="cuda", dtype=torch.int32)

    sentinel = 7.5
    out = torch.full((M_total, I), sentinel, device="cuda", dtype=torch.bfloat16)
    invoke_gluon_act_and_mul(gateup, out, expert_ids=expert_ids,
                             expert_step=expert_step, activation="silu")

    gate = gateup[:, :I].float()
    up = gateup[:, I:].float()
    ref_full = (F.silu(gate) * up).to(torch.bfloat16)
    expected = torch.where(
        (expert_ids != -1).view(-1, 1),
        ref_full,
        torch.full_like(ref_full, sentinel),
    )
    torch.testing.assert_close(out, expected, atol=2e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
#  moe_sum_reduce
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("M", [16, 256])
@pytest.mark.parametrize("top_k", [2, 8])
@pytest.mark.parametrize("D", [512, 4096])
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_moe_sum_reduce(M, top_k, D, scale):
    torch.manual_seed(0)
    x = torch.randn(M, top_k, D, device="cuda", dtype=torch.bfloat16) * 0.1

    out = torch.empty(M, D, device="cuda", dtype=torch.bfloat16)
    invoke_gluon_moe_sum_reduce(x, out, routed_scaling_factor=scale)

    ref = (x.float().sum(dim=1) * scale).to(torch.bfloat16)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=1e-2)


# --------------------------------------------------------------------------- #
#  GEMM (gluon_fused_moe_kernel)
# --------------------------------------------------------------------------- #


def _torch_moe_gemm_ref(A, W, sorted_ids, expert_ids, npp, num_valid, top_k,
                        block_m, mul_routed_weight, topk_weights, dtype):
    """Vectorised reference: one matmul per expert, masked by routing."""
    M, D = A.shape
    E, two_I, _ = W.shape
    out = torch.zeros(num_valid, two_I, dtype=dtype, device=A.device)

    npp_v = int(npp.item())
    flat_ti = topk_weights.flatten() if mul_routed_weight else None

    # Build (sorted_idx, valid_mask, expert_id_for_token) for every entry of
    # sorted_ids that is in-range.
    sorted_ids_h = sorted_ids[:npp_v]
    expert_ids_h = expert_ids[:npp_v // block_m]
    # Repeat each block's expert id `block_m` times.
    expert_per_slot = expert_ids_h.repeat_interleave(block_m)
    valid = (sorted_ids_h < num_valid) & (expert_per_slot != -1)
    valid_token = sorted_ids_h[valid].long()
    valid_expert = expert_per_slot[valid].long()
    src_row = valid_token // top_k

    # One matmul per expert.
    for e in range(E):
        mask = valid_expert == e
        if mask.any():
            rows = src_row[mask]
            tids = valid_token[mask]
            res = (A[rows].float() @ W[e].float().T)
            if mul_routed_weight:
                res = res * flat_ti[tids].float().unsqueeze(-1)
            out[tids] = res.to(dtype)
    return out


@pytest.mark.parametrize(
    "E,D,I,top_k,bs",
    [
        (4, 256, 256, 2, 64),
        (8, 512, 512, 4, 128),
        (16, 1024, 512, 4, 256),
    ],
)
@pytest.mark.parametrize("mul_routed_weight", [False, True])
def test_fused_moe_gemm(E, D, I, top_k, bs, mul_routed_weight):
    BM, BN, BK = 128, 128, 64
    if D % BK != 0:
        pytest.skip(f"D={D} not divisible by BK={BK}")

    torch.manual_seed(0)
    A = torch.randn(bs, D, device="cuda", dtype=torch.bfloat16) * 0.1
    W = torch.randn(E, 2 * I, D, device="cuda", dtype=torch.bfloat16) * 0.05
    sc = torch.softmax(torch.randn(bs, E, device="cuda"), dim=-1)
    tw, ti = torch.topk(sc, top_k)
    ti = ti.to(torch.int32)
    tw = tw.to(torch.float32)

    sorted_ids, expert_ids, npp = gluon_moe_align_block_size(ti, BM, E)

    C = torch.zeros(bs * top_k, 2 * I, dtype=torch.bfloat16, device="cuda")
    cfg = {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK,
           "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2}

    invoke_gluon_fused_moe_kernel(
        A, W, C, tw, ti, sorted_ids, expert_ids, npp,
        mul_routed_weight=mul_routed_weight, top_k=top_k, config=cfg,
        compute_type=gl.bfloat16, c_sorted=False, filter_expert=True,
    )

    ref = _torch_moe_gemm_ref(
        A, W, sorted_ids, expert_ids, npp, bs * top_k, top_k, BM,
        mul_routed_weight, tw, torch.bfloat16,
    )

    # bf16 GEMM accumulator round-off plus differences in summation order
    # between the MFMA pipeline and torch's reference.
    torch.testing.assert_close(C.float(), ref.float(), atol=5e-2, rtol=5e-2)


# --------------------------------------------------------------------------- #
#  Pipeline integration: chain all 5 sublocks
# --------------------------------------------------------------------------- #


def test_moe_pipeline_integration():
    """Run topk -> align -> gate_up GEMM -> silu -> down GEMM -> reduce
    using only the Gluon kernels, and compare against a torch reference for
    the full pipeline. This is the smallest end-to-end correctness check
    that exercises every sublock once."""
    torch.manual_seed(0)
    E, D, I, top_k, bs = 8, 512, 512, 2, 64
    BM, BN, BK = 128, 128, 64
    cfg = {"BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK,
           "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 2}

    x = torch.randn(bs, D, device="cuda", dtype=torch.bfloat16) * 0.1
    w1 = torch.randn(E, 2 * I, D, device="cuda", dtype=torch.bfloat16) * 0.05
    w2 = torch.randn(E, D, I, device="cuda", dtype=torch.bfloat16) * 0.05
    logits = torch.randn(bs, E, device="cuda", dtype=torch.float32)

    # Stage 1: topk + softmax.
    tw = torch.empty(bs, top_k, device="cuda", dtype=torch.float32)
    ti = torch.empty(bs, top_k, device="cuda", dtype=torch.int32)
    invoke_gluon_topk_softmax(logits, tw, ti, renormalize=True)

    # Stage 2: align.
    sorted_ids, expert_ids, npp = gluon_moe_align_block_size(ti, BM, E)

    # Stage 3: gate_up GEMM.
    C1 = torch.zeros(bs * top_k, 2 * I, dtype=torch.bfloat16, device="cuda")
    invoke_gluon_fused_moe_kernel(
        x, w1, C1, tw, ti, sorted_ids, expert_ids, npp,
        mul_routed_weight=False, top_k=top_k, config=cfg,
        compute_type=gl.bfloat16, c_sorted=False, filter_expert=True,
    )

    # Stage 4: silu_and_mul.
    C2 = torch.empty(bs * top_k, I, dtype=torch.bfloat16, device="cuda")
    invoke_gluon_act_and_mul(C1, C2, activation="silu")

    # Stage 5: down GEMM (with mul_routed_weight, c_sorted=False).
    C3_flat = torch.zeros(bs * top_k, D, dtype=torch.bfloat16, device="cuda")
    invoke_gluon_fused_moe_kernel(
        C2, w2, C3_flat, tw, ti, sorted_ids, expert_ids, npp,
        mul_routed_weight=True, top_k=1, config=cfg,
        compute_type=gl.bfloat16, c_sorted=False, filter_expert=True,
    )

    # Stage 6: moe_sum_reduce.
    C3 = C3_flat.view(bs, top_k, D)
    out = torch.empty(bs, D, dtype=torch.bfloat16, device="cuda")
    invoke_gluon_moe_sum_reduce(C3, out, routed_scaling_factor=1.0)

    # Reference: same pipeline but in plain torch.
    sm = torch.softmax(logits.double(), dim=-1).to(torch.float32)
    ref_w_full, ref_i = torch.topk(sm, top_k)
    ref_w = ref_w_full / ref_w_full.sum(dim=-1, keepdim=True)
    ref_i = ref_i.to(torch.int64)

    ref_out = torch.zeros(bs, D, dtype=torch.float32, device="cuda")
    for m in range(bs):
        for k in range(top_k):
            e = int(ref_i[m, k].item())
            w_e = w1[e].float()
            gate_up = x[m].float() @ w_e.T               # (2*I,)
            gate, up = gate_up[:I], gate_up[I:]
            inter = (F.silu(gate) * up)                  # (I,)
            down = inter @ w2[e].float().T               # (D,)
            ref_out[m] += down * ref_w[m, k].float()

    torch.testing.assert_close(out.float(), ref_out, atol=0.2, rtol=0.1)
