# Buffer-soffset split: investigation, findings, and plan

This document records what we tested, what we measured, what works,
what does not, and the proposed path forward for the AMD buffer
soffset-split optimization on `triton-buffer-soffset-split`.

## tl;dr

- The optimization works: it lifts a wave-uniform additive component of
  a buffer-op offset into the `soffset` SGPR slot, freeing per-lane VGPR
  adds in the hot loop. We measured **-12 % to -23 %** inner-loop
  instruction reductions and **-53 % to -90 %** `v_add_u32` removals on
  three real kernels.
- Wall-clock TFLOPs barely move on these kernels because they are
  MFMA + LDS bound, so the freed VALU cycles are slack on the
  non-critical pipe. The pass buys VALU headroom that future scheduling
  work can convert to TFLOPs; on its own it is performance-neutral on
  the kernels we benchmarked.
- The current PR placement at HEAD (MLIR-level, in
  `TritonAMDGPUTransforms/ConvertToBufferOps.cpp`) does not actually
  trigger on any of the gluon-emitted kernels in the test suite, for
  three independent reasons documented below. Adding the pass to
  `gluon_to_ttgir` is necessary but not sufficient.
- The earlier LLVM-level placement (commits `72883464dd + 147848a3f2`,
  introduced + hardened) does trigger on these kernels and produces the
  measured codegen reductions. Two follow-on extensions land additional
  splits (FA goes from 0/24 to 20/24 buffer ops splittable).

## What we tested, in order

Each section below corresponds to an experiment we actually ran. Every
claim has an artifact path and a reproducibility recipe.

### 1. Sanity that the LLVM-level placement actually fires

Branch `experimental` was created at the two original LLVM-level
commits:

```
git checkout -b experimental 147848a3f2fac6e4ee04eb2aacc8756002cf755c
# git log -2 --oneline:
# 147848a3f2 amd: harden buffer soffset split lowering
# 72883464dd amd: split uniform buffer-load offsets during lowering
```

Verification: dumped AMDGCN for `mxfp4_aiter_opt_inner_aligned_unroll4`
under "soffset-split ON" vs "OFF" snapshots and parsed every
`raw.ptr.buffer.load.*` intrinsic's soffset operand:

```
on : total=111  non-zero soffsets=101  top10=[('0', 10), ('256', 9), ('128', 8),
                                              ('%855', 8), ('%1103', 8), ...]
off: total=120  non-zero soffsets= 16  top10=[('0', 104), ...]
```

91 % of buffer ops route through soffset with the pass on, 13 % without.
The literal K-stride bumps (128, 256) and dynamic uniform SGPRs
(`%855`, `%1103`, ...) confirm exactly the pattern your PR description
targets.

Artifact: `gluon_mxfp4/build/mxfp4_aiter_opt_inner_aligned_unroll4_{on,off}/`.

### 2. v* MXFP4 GEMM perf vs base, idle GPU

Bench script: `benchmarking_mxfp4_kerns/gemm/_bench.py`. Two timed
runs per branch, averaged. Idle GPU (verified `rocm-smi` shows 0 % util
before each run).

Per-shape geomean(exp/base) on 19 v* kernels x 5 shapes (95 cells):

| shape                  | exp_v1 | exp_v2 |
| ---------------------- | -----: | -----: |
| M=4096 N=512  K=7168   | 0.997  | 0.985  |
| M=2048 N=2048 K=7168   | 1.002  | 0.992  |
| M=1024 N=8192 K=7168   | 1.002  | 1.012  |
| M=8192 N=1024 K=7168   | 1.007  | 1.013  |
| M=256  N=256  K=7168   | 0.998  | 0.987  |

Overall: 1.0010 (exp_v1) and 0.9976 (exp_v2). Both well within the
noise floor of these benchmarks. **No wall-clock signal on these
kernels.**

Artifacts: `benchmarking_mxfp4_kerns/perf_base_idle.csv`,
`perf_experimental_idle.csv`, `perf_experimental_ext12_idle.csv`,
report at `perf_experimental_vs_base_idle.md`.

### 3. Flash Attention perf vs base, idle GPU

Same harness, FA STANDARD sweep (13 configs):

| config                      | base TFLOPs | exp_v1 | exp_v2 |
| --------------------------- | ----------: | -----: | -----: |
| ... 13 rows ...             |             |        |        |
| AVERAGE                     | 719.4       | 719.4  | 721.8  |

Per-config absolute deltas <= 1.5 % in both directions, average +0.33 %
in exp_v2's favor. Within noise.

Artifact: `/tmp/fa_base_idle.log`, `/tmp/fa_ext2_bench.log`.

### 4. Hot-loop instruction analysis

Located the `Inner Loop Header (Depth=1)` block in each AMDGCN dump and
counted instruction categories within just that block. Tooling at
`/tmp/_hotloop_analyze.py`. Same M=4096 N=512 K=7168 shape across all
configs:

```
kernel                        src         loop   tot buf sof imm add mfma  ds  vgpr sgpr
v3                           base      .LBB0_5   186  28   0   0  22   32  20   136   31
v3                         exp_v1      .LBB0_5   186  28   0   0  22   32  20   136   32
v3                         exp_v2      .LBB0_5   187  28   0   0  22   32  20   136   32

v3_pingpong                  base      .LBB0_4   182  28   0   0  31   32  20   136   43
v3_pingpong                exp_v1      .LBB0_4   166  28  28   0   3   32  20   154   46
v3_pingpong                exp_v2      .LBB0_4   160  28  28   0   3   32  20   144   47

v3_pingpong_triple_lds       base      .LBB0_4   234  28   0   0  35   32  26   162   48
v3_pingpong_triple_lds     exp_v1      .LBB0_4   180  28  28   0   7   32  26   160   48
v3_pingpong_triple_lds     exp_v2      .LBB0_4   180  28  28   0   7   32  26   162   48

v4_addr_simpl                base      .LBB0_5   182  28   0   0   0   32  20   140   52
v4_addr_simpl              exp_v1      .LBB0_5   181  28   0  10   0   32  20   124   44
v4_addr_simpl              exp_v2      .LBB0_5   181  28   0  10   0   32  20   142   53

FA base                       .LBB0_19  558   8   0  15   4   64  96   236   96
FA exp_v1 (commits only)      .LBB0_19  558   8   0  15   4   64  96   236   96
FA exp_v2 (commits+Ext1+2)    .LBB0_19  542   8   8   7   8   64  96   238   96
```

Highlights (base -> exp_v2):
- `v3_pingpong`            : 182 -> 160 instr (-12.1 %), v_add 31 -> 3 (-90 %)
- `v3_pingpong_triple_lds` : 234 -> 180 instr (-23.1 %), v_add 35 -> 7 (-80 %)
- FA main attn loop        : 558 -> 542 instr (-2.9 %), v_add 15 -> 7 (-53 %), v_mul 10 -> 2 (-80 %)
- `v4_addr_simpl`          : -16 VGPRs (140 -> 124) on exp_v1 (10 imm-soffsets from unrolled K)

Why wall-clock didn't move: every reduction is in `v_add_u32`, `v_mul`,
`v_mov`. The compute work (`v_mfma_*`, `ds_*`) is unchanged. The freed
VALU cycles run in parallel with the MFMAs and LDS loads on a different
functional unit, so they were already slack.

Artifacts: `benchmarking_mxfp4_kerns/asm/{base_v2,experimental,exp_v2,head,head_p1}/v*/mxfp4_gemm_kernel.amdgcn`
+ `asm/fa_*/gluon_attn_fwd*.amdgcn`. Tooling: `/tmp/_hotloop_analyze.py`,
`/tmp/_hotloop_fa.py`.

### 5. Diagnostic instrumentation

LLVM-level pass (in `OffsetUniformitySplit.cpp`):
`TRITON_AMD_SOFFSET_SPLIT_DEBUG=1` enables one line per buffer-op
emission with function name, leaf classification (uniform / per-lane),
root-cause trace for non-uniform leaves, and a SPLIT/NO-SPLIT decision.
Off by default; codegen-identical when off (verified by SHA-256 match
with the un-instrumented build).

MLIR-level pass (in `ConvertToBufferOps.cpp`):
`TRITON_AMD_BUFFER_OPS_DEBUG=1` enables the same on
`splitHighLevelBufferOffset`.

### 6. Extension 1: recognize `rocdl.ds_bpermute` with uniform index

```cpp
if (auto bperm = dyn_cast<ROCDL::DsBpermuteOp>(def)) {
  if (isUniform(bperm.getIndex())) return true;
  recordWhy(v, "ds_bpermute with non-uniform index");
  return false;
}
```

Verification: per-kernel split rate before/after Ext 1 across FA, gk
f16 GEMM, v3 MXFP4, unroll4 - no change on any of them (those kernels
use bpermute as lane swizzles with per-lane indices). Diagnostic now
classifies precisely (`ds_bpermute with non-uniform index` vs
`unrecognized op`).

Land it anyway - safe and waiting for kernels that use bpermute as a
readfirstlane-style broadcast.

### 7. Extension 2: descend through `mul/shl` with literal-or-uniform multiplier

The literal-only version proposed originally didn't fire on v3 muls
because their operands are both `llvm.extractvalue` of struct-packed
kernel args (uniform but not literal). Generalised to also descend when
one operand is wave-uniform.

`LeafDescriptor` now carries `APInt constMul` plus
`SmallVector<Value> dynMuls`; materialisation is lazy, on the commit
path, so no orphan IR on bail.

Verification, before/after Ext 2 (on top of Ext 1):

| kernel                         | split rate before -> after Ext 2 | per-call detail |
| ------------------------------ | -------------------------------- | --------------- |
| FA (`gluon_attn_fwd`)          | 0/24 -> 20/24                    | each lifts 1 uniform; per-lane reduced from xor+mul to xor only |
| `gk_f16_gemm_4warp`            | 0/64 -> 0/64                     | uniform leaves grew 0 -> 2 but both literal-zero, dropped       |
| v3 MXFP4 GEMM                  | 16/72 -> 16/72                   | C-stores lift 2 uniforms each (was 1)                           |
| unroll4                        | 100/112 -> 100/112               | 16 SPLITs lift 2 uniforms each (was 1)                          |

Codegen change confirmed by per-kernel inner-loop instruction count
(see Section 4).

### 8. Validation against HEAD (sanketp/buffer-soffset-split, MLIR-level placement)

After committing the LLVM-level changes, we switched back to HEAD,
rebuilt, and re-ran the same dump + analyzer. **Every AMDGCN file is
byte-identical to base.** The only difference is one debug-path string:

```
< .file 2 "/workspace/triton-buffer-soffset-base/python/triton/language" "standard.py"
> .file 2 "/workspace/triton-buffer-soffset-split/python/triton/language" "standard.py"
```

This is the entire diff, repeated for all 10 v* kernels and FA. SHA-256
confirmed.

Reading: the MLIR-level pass at HEAD does not fire on any gluon kernel
in the test suite.

Artifact: `benchmarking_mxfp4_kerns/asm/head/v*/mxfp4_gemm_kernel.amdgcn`
+ `perf_head_vs_experimental.md`.

### 9. Option 1 fix: add `add_convert_to_buffer_ops` to `gluon_to_ttgir`

Branch `experimental-mlir-fix` (HEAD = `cebd2b2bc8`).

```python
# third_party/amd/backend/compiler.py, gluon_to_ttgir
if knobs.amd.use_buffer_ops:
    amd.passes.ttgpuir.add_convert_to_buffer_ops(
        pm, options.arch, knobs.amd.use_buffer_atomics,
        knobs.amd.buffer_ops_analyze_small_tensor_range,
    )
    amd.passes.ttgpuir.add_optimize_buffer_op_ptr(pm)
```

Verification with `TRITON_AMD_BUFFER_OPS_DEBUG=1`:

For `v3_pingpong` (9 walker invocations):

```
[mlir-soffset]  fn=mxfp4_gemm_kernel  offset_root=arith.addi  u=1  pl=2  pl_kind=tt.broadcast  pl_kind=tt.broadcast
[mlir-soffset]    bail: per-lane leaf not provably non-negative
... 6x identical ...
```

For FA (44 invocations):

```
[mlir-soffset]  fn=...attn_fwd_inner...  offset_root=arith.addi  u=0  pl=2  pl_kind=tt.broadcast  pl_kind=tt.broadcast
... 44x identical ...
```

Hot-loop counts after the fix: bit-identical to base (SHA-256 match for
all 10 v* kernels and FA). The fix is necessary (without it the pass
never sees gluon ops at all) but not sufficient.

Two upstream blockers come into play:

**Blocker A (FA-shape kernels): walker stops at top-level `tt.broadcast`.**
Pristine walker descends only through `arith::AddIOp` and lifts
`tt::SplatOp(uniform)`. Gluon-emitted offsets are typically
`addi(broadcast(M_part), broadcast(K_part))`; the broadcast hides the
inner addition tree. Walker finds 0 uniform leaves, can never split.

**Blocker B (v3_pingpong-shape kernels): non-negativity safety guard.**
Walker correctly identifies `pid_m * BLOCK_M` as a uniform leaf 6 / 9
times, but the safety check `isProvenNonNegative(perLaneLeaves)`
rejects every one because the per-lane `tt.broadcast` values can't be
proven `>= 0` from the integer-range solver alone:

```cpp
// AMD raw buffer ops bound-check `voffset` without including `soffset`.
auto nonNegative = [&](Value v) { return isProvenNonNegative(v, solver); };
if (!llvm::all_of(perLaneLeaves, nonNegative))
  return {offset, Value()};
```

This is correct in principle; the LLVM-level pass at the experimental
branch does NOT have this guard and just splits unconditionally, which
is why its codegen win shows up.

Artifact: `benchmarking_mxfp4_kerns/perf_head_with_pipeline_fix.md` and
`asm/{head_p1,fa_head_p1}/`.

## Summary of where the pass fires today

| placement                                       | kernels covered     | effective |
| ----------------------------------------------- | ------------------- | :-------: |
| MLIR `ConvertToBufferOps`, `make_ttgir`         | tt.load -> buffer   |    yes    |
| MLIR `ConvertToBufferOps`, `gluon_to_ttgir`     | gluon buffer ops    | no (Blockers A, B) |
| MLIR walker descends `tt.broadcast` (in-place)  | gluon broadcast off | breaks `verify_each` |
| LLVM-level `OffsetUniformitySplit`              | everything by then  |    yes    |
| LLVM-level + Ext 1 (`ds_bpermute`)              | future bpermute use |    safe   |
| LLVM-level + Ext 2 (`mul/shl` descent)          | gluon FA + v3 lifts | yes       |

## Plan forward

### Immediate (low risk, lands wins now)

1. **Reintroduce the LLVM-level pass** alongside the MLIR-level
   placement. The two operate at different IR levels on different
   patterns and don't conflict.

   - Cherry-pick commits `72883464dd` and `147848a3f2` onto HEAD.
   - Cherry-pick the Ext 1 + Ext 2 extension on the `experimental`
     branch (commit `1b0bca340f`).
   - The MLIR-level pass continues handling tt.load-converted buffer
     ops in `make_ttgir`. The LLVM-level pass picks up everything that
     survives to scalar IR (gluon kernels, atomics, etc.).
   - Hot-loop reductions documented in section 4 land for v3_pingpong*
     and FA.

2. **Add the gluon-pipeline call** (`experimental-mlir-fix` cebd2b2bc8).
   No effect today on its own (Blockers A and B), but sets up the MLIR
   pass to fire on gluon ops once Blockers A and B are addressed in
   the future. Can land independently or together with item 1.

3. **Land the env-gated diagnostics** in both passes. Off by default,
   codegen-identical, so risk-free; makes future debugging much faster.

### Medium-term (unblocks gluon kernels at MLIR level)

4. **Plan-then-commit walker** for `ConvertToBufferOps.cpp`. The walker
   gathers a `LeafDescriptor` plan with no IR creation; the top-level
   `splitHighLevelBufferOffset` decides go/no-go after seeing the
   complete plan; only on commit does it materialise. Resolves
   Blocker A by allowing safe descent through `tt.broadcast` /
   `tt.expand_dims` / `arith.muli` with uniform / `arith.shli`.

5. **Either** improve the integer-range solver to propagate
   non-negativity through `tt.broadcast` / `tt.expand_dims` / `divsi`
   chains (Blocker B), OR tie the safety guard to the kernel's masking
   mode. The LLVM-level `BufferEmitter::fillCommonArgs` already wraps
   the per-lane offset in `select(pred, voffset, OOB)`, so for
   unmasked accesses there is nothing to OOB-bypass and the guard
   could relax to `(masked || provenNonNeg)`.

### Long-term (cracks the modular MXFP4 scale formula)

6. **Range / divisibility-driven simplification of
   `divsi(addi(splat_u, per_lane_bounded), const_c)` -> `splat(u/c)`**
   when range analysis proves the per-lane component fits a single
   quotient bucket. This is the only thing left that would make v* MXFP4
   scale loads (NOT just C-stores) actually split. It needs structural
   divisibility tracking that the current `IntegerRangeAnalysis`
   doesn't carry; the right place is probably `AxisInfoAnalysis` which
   already tracks divisibility for pointer alignment.

## 2026-05-12 frontend-only validation

Validation branch: `experimental-mlir-fix` at `ff9c639589`.

This run checked whether the frontend-only `ConvertToBufferOps.cpp`
change reproduces the experimental branch's assembly effect without an
LLVM-level pass. It does not yet match experimental. The pass now
partially moves uniform offset work into `soffset` for the pingpong GEMM
variants, but remains too conservative for the full experimental effect
and is still base-like for `v4_addr_simpl` and FA.

| kernel | experimental target | current frontend-only result |
| ------ | ------------------- | ---------------------------- |
| `v3` | `total=186`, `soffset=0`, `v_add=22` | matches: `total=186`, `soffset=0`, `v_add=22` |
| `v3_pingpong` | `total=160`, `soffset=28`, `v_add=3` | partial: `total=172`, `soffset=20`, `v_add=11` |
| `v3_pingpong_triple_lds` | `total=180`, `soffset=28`, `v_add=7` | partial: `total=190`, `soffset=20`, `v_add=15` |
| `v4_addr_simpl` | `total=181`, `imm=10`, `v_add=0` | base-like: `total=182`, `imm=0`, `v_add=0` |
| FA main loop | `total=542`, `soffset=8`, `v_add=8`, `v_mul=2` | base-like: `total=558`, `soffset=0`, `v_add=15`, `v_mul=10` |

Full-kernel buffer-op split counts for the GEMM assembly dumps:

| kernel | buffer ops | nonzero `soffset` | zero `soffset` | nonzero immediate |
| ------ | ---------: | ----------------: | -------------: | ----------------: |
| `v3` | 72 | 0 | 72 | 0 |
| `v3_pingpong` | 72 | 40 | 32 | 0 |
| `v3_pingpong_triple_lds` | 100 | 20 | 80 | 0 |
| `v4_addr_simpl` | 72 | 0 | 72 | 0 |

Diagnostics explain the gap:

- `v3_pingpong*`: the frontend pass splits some `divui` / `remui`
  forms, but still bails on `tt.make_range` / `arith.remsi` leaves
  because the per-lane component is not proven non-negative.
- FA: the hot loop is still effectively base-like. The debug trace shows
  the same non-negativity bailouts in inner functions, and the top-level
  FA offset still exposes no useful uniform leaves for the current
  frontend walker.

Performance was also run, but the machine was not idle: `rocm-smi`
reported all GPUs at 100% utilization before the sweep. The GEMM sweep
completed `95/95`, but the resulting CSV is not comparable to the
idle-GPU numbers in this file. The contended run measured roughly 3x
slower geomean than the saved idle CSVs, which should be treated as
machine contention rather than a compiler regression. FA quick completed
at `583.9 TFLOPS / 1.412 ms`, also under contention.

Artifacts:

- `benchmarking_mxfp4_kerns/asm/experimental_mlir_fix_frontend/`
- `benchmarking_mxfp4_kerns/perf_experimental_mlir_fix_frontend.csv`
- `benchmarking_mxfp4_kerns/asm/fa_experimental_mlir_fix_frontend/`
- `benchmarking_mxfp4_kerns/fa_experimental_mlir_fix_frontend_quick.json`

## Reproducibility

```bash
# Branches:
# - experimental (LLVM-level + Ext 1 + Ext 2): 1b0bca340f
# - experimental-mlir-fix (gluon-pipeline addition): cebd2b2bc8
# - sanketp/buffer-soffset-split (HEAD): 89f627d338
# - triton-buffer-soffset-base (no pass): e9ac8e40a9

cd /home/sanketp/work/triton-buffer-soffset-split
git checkout <branch>
docker exec sanketp_triton_dev bash -c '
  cd /workspace/triton-buffer-soffset-split/build/cmake.linux-x86_64-cpython-3.10
  ninja triton
'

# v* GEMM AMDGCN dump
HIP_VISIBLE_DEVICES=7 \
  PYTHONPATH=/workspace/triton-buffer-soffset-split/python:/workspace/aiter:/workspace \
  python /workspace/benchmarking_mxfp4_kerns/gemm/_dump_asm.py \
    --version v3_pingpong --M 4096 --N 512 --K 7168 \
    --out /tmp/v3p_dump

# Hot-loop counts
python3 /tmp/_hotloop_analyze.py        # v* table
python3 /tmp/_hotloop_fa.py             # FA table

# v* GEMM bench (idle GPU only)
python /workspace/benchmarking_mxfp4_kerns/gemm/_bench.py \
  --label <label> --out <out>.csv --warmup 10 --iters 50

# FA bench (idle GPU only)
cd /workspace/gluon-kernels/kernels/cdna4/fa
python f16_fa_gfx950.py                  # full STANDARD sweep

# Diagnostics
TRITON_AMD_SOFFSET_SPLIT_DEBUG=1 ...     # LLVM-level pass
TRITON_AMD_BUFFER_OPS_DEBUG=1 ...        # MLIR-level pass
```

## Pointers to reports

- `benchmarking_mxfp4_kerns/perf_hotloop_codediff.md` -- inner-loop
  instruction reductions per kernel.
- `benchmarking_mxfp4_kerns/perf_ext12_summary.md` -- before/after
  each of Ext 1 and Ext 2 individually.
- `benchmarking_mxfp4_kerns/perf_head_vs_experimental.md` -- HEAD's
  MLIR-level placement vs experimental's LLVM-level placement,
  byte-identical-to-base proof.
- `benchmarking_mxfp4_kerns/perf_head_with_pipeline_fix.md` -- result of
  Option 1 (add pass to gluon pipeline) and the two upstream blockers.
- `benchmarking_mxfp4_kerns/perf_experimental_vs_base_idle.md` -- v* GEMM
  geomean comparison, idle-GPU clean window.
- `benchmarking_mxfp4_kerns/perf_compare_experimental.md` and
  `perf_compare.md` -- earlier under-contention runs (kept for
  context; superseded by the idle-GPU report).
