"""
Gluon SwiGLU (silu_and_mul) and GeGLU (gelu_and_mul) for the MOE pipeline.

Replaces sglang's ``act_and_mul_kernel`` for the bf16 path. One CTA per
(token * topk) row; each CTA streams ``BLOCK_DIM``-wide chunks across the
half-hidden dim.

Layout:
    Input  ``gateup_output[M_total, 2 * I]`` bf16
    Output ``down_input[M_total, I]`` bf16
    For each row m:
        gate = gateup_output[m, 0:I]
        up   = gateup_output[m, I:2*I]
        down_input[m, :] = activation(gate.float()) * up

Optionally skips rows whose expert id is ``-1`` (used when the routing
dispatcher zeros out experts that are not on this rank).
"""

from __future__ import annotations

import os
from typing import Optional

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


_USE_GLUON_ACTIVATION_ENV = "SGLANG_USE_GLUON_ACTIVATION"


def use_gluon_activation() -> bool:
    if not _GLUON_AVAILABLE:
        return False
    val = os.environ.get(_USE_GLUON_ACTIVATION_ENV, "0").lower()
    return val in ("1", "true", "yes", "on")


if _GLUON_AVAILABLE:

    @gluon.jit
    def _sigmoid(x):
        # gl.sigmoid is not in the Gluon language module; build from exp.
        return 1.0 / (1.0 + gl.exp(-x))

    @gluon.jit
    def _tanh(x):
        return 2.0 * _sigmoid(2.0 * x) - 1.0

    @gluon.jit
    def gluon_act_and_mul_kernel(
        gateup_ptr,        # [M_total, 2 * I]
        out_ptr,           # [M_total, I]
        expert_ids_ptr,    # [M_total / expert_step] int32, may be None
        I,                 # half-hidden dim (compile-time-known via specialization)
        gateup_stride_m,   # stride along M_total in elements
        out_stride_m,      # stride along M_total in elements
        expert_step: gl.constexpr,
        BLOCK_DIM: gl.constexpr,
        ACTIVATION: gl.constexpr,   # "silu" or "gelu"
        FILTER_EXPERT: gl.constexpr,
        NUM_WARPS: gl.constexpr,
    ):
        pid = gl.program_id(axis=0)

        if FILTER_EXPERT:
            expert_id = gl.load(expert_ids_ptr + pid // expert_step)
            if expert_id == -1:
                return

        # 1D blocked layout for the dim axis. 8 elements per thread = 128 bits
        # for bf16 keeps coalescing wide and matches the buffer_load chunk size.
        layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[8],
            threads_per_warp=[64],
            warps_per_cta=[NUM_WARPS],
            order=[0],
        )

        gate_base = gateup_ptr + pid.to(gl.int64) * gateup_stride_m
        up_base = gate_base + I
        out_base = out_ptr + pid.to(gl.int64) * out_stride_m

        for start in range(0, I, BLOCK_DIM):
            offs = start + gl.arange(0, BLOCK_DIM, layout=layout)
            mask = offs < I

            gate = gl.load(gate_base + offs, mask=mask, other=0.0)
            up = gl.load(up_base + offs, mask=mask, other=0.0)

            x = gate.to(gl.float32)
            if ACTIVATION == "silu":
                act = x * _sigmoid(x)
            elif ACTIVATION == "gelu":
                kAlpha = 0.7978845608028654
                act = 0.5 * x * (1.0 + _tanh(kAlpha * (x + 0.044715 * x * x * x)))
            else:
                gl.static_assert(False, "unsupported activation")
                act = x

            res = act.to(gateup_ptr.type.element_ty) * up
            gl.store(out_base + offs, res.to(out_ptr.type.element_ty), mask=mask)


def invoke_gluon_act_and_mul(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    *,
    expert_ids: Optional[torch.Tensor] = None,
    expert_step: int = 1,
    activation: str = "silu",
    block_dim: int = 1024,
    num_warps: int = 4,
) -> None:
    """Launch the Gluon activation kernel.

    Args:
        gateup_output: shape ``(M_total, 2 * I)`` bf16/fp16.
        down_input: shape ``(M_total, I)`` same dtype.
        expert_ids: optional ``int32`` tensor; row m is skipped when
            ``expert_ids[m // expert_step] == -1``.
        expert_step: stride for indexing ``expert_ids`` (typically 1 for
            unsorted layout, ``BLOCK_SIZE_M`` for sorted layout).
        activation: ``"silu"`` or ``"gelu"``.
        block_dim: column-tile size per CTA iteration.
    """
    assert _GLUON_AVAILABLE, "Gluon is not available in this Triton build"
    assert gateup_output.dim() == 2 and down_input.dim() == 2
    assert gateup_output.shape[0] == down_input.shape[0]
    assert gateup_output.shape[1] == 2 * down_input.shape[1]
    assert gateup_output.dtype == down_input.dtype

    M_total, two_I = gateup_output.shape
    I = two_I // 2
    grid = (M_total,)

    gluon_act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        expert_ids if expert_ids is not None else gateup_output,  # ignored when not FILTER_EXPERT
        I,
        gateup_output.stride(0),
        down_input.stride(0),
        expert_step=expert_step,
        BLOCK_DIM=block_dim,
        ACTIVATION=activation,
        FILTER_EXPERT=expert_ids is not None,
        NUM_WARPS=num_warps,
        num_warps=num_warps,
    )
