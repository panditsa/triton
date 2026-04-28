// RUN: triton-opt %s -split-input-file --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch=gfx950 | FileCheck %s

// Tests for the small-alphabet cyclic-index recognition in
// MemDescIndexOpConversion. The conversion folds the per-iteration
// `mul + applyPadding` chain into precomputed offsets selected by the index
// when the index is provably restricted to {0, ..., N-1} with N <= NUM_BUFFERS.
//
// Patterns recognized:
//   1. arith.constant c, with c in [0, NUM_BUFFERS).
//      => single llvm.mlir.constant offset, no select.
//   2. arith.remui x, NUM_BUFFERS.
//      => N-1 selects over precomputed offsets.
//   3. arith.andi x, (N-1) for N a power of 2 and N <= NUM_BUFFERS.
//      => N-1 selects over precomputed offsets.
//
// All other index shapes fall back to the original `mul + applyPadding`
// chain, which is exercised by the negative tests at the end.
//
// Stride per buffer in the padded layout below is 64 * 64 = 4096 elements;
// padding is `(raw >> log2(128)) << log2(4)` so for raw == 4096 the pad is
// (4096 >> 7) << 2 = 32 << 2 = 128, giving a per-buffer offset of 4224.

// =============================================================================
// 1. Single-element alphabet: arith.constant index
// =============================================================================

// CHECK-LABEL: const_index_padded_buf0
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Buffer 0 of a 2-buffer padded layout: offset = 0.
  tt.func @const_index_padded_buf0(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>) {
    %c0_i32 = arith.constant 0 : i32

    // Pass must NOT emit any of the original chain ops anywhere in the body.
    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.lshr
    // CHECK-NOT: llvm.shl
    // CHECK-NOT: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%c0_i32] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: const_index_padded_buf1
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Buffer 1: raw = 4096, pad = 128, total = 4224. The recognizer folds the
  // entire chain into a single i32 constant `4224` used as the GEP offset.
  tt.func @const_index_padded_buf1(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>) {
    %c1_i32 = arith.constant 1 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.lshr
    // CHECK-NOT: llvm.shl
    // CHECK-NOT: llvm.select
    // CHECK: llvm.mlir.constant(4224 : i32) : i32
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%c1_i32] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: const_index_swizzled_buf1
#shared_sz = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Swizzled (no padding) layout: offset = 1 * 4096 = 4096.
  tt.func @const_index_swizzled_buf1(%arg0: !ttg.memdesc<2x64x64xf16, #shared_sz, #smem, mutable>) {
    %c1_i32 = arith.constant 1 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.select
    // CHECK: llvm.mlir.constant(4096 : i32) : i32
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%c1_i32] : !ttg.memdesc<2x64x64xf16, #shared_sz, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_sz, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 2. Two-element alphabet: arith.andi(x, 1)  (canonical form of `x % 2`)
// =============================================================================

// -----

// CHECK-LABEL: andi_mask1_padded
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Power-of-2 mask 1 -> alphabet {0, 1}. Expect a single select between the
  // two precomputed offsets (0 and 4224), no `mul`/`lshr`/`shl`.
  tt.func @andi_mask1_padded(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c1_i32 = arith.constant 1 : i32
    %idx = arith.andi %k, %c1_i32 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.lshr
    // CHECK-NOT: llvm.shl
    // CHECK-DAG: llvm.icmp "eq"
    // CHECK-DAG: llvm.mlir.constant(4224 : i32) : i32
    // CHECK: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: andi_mask1_swizzled
#shared_sz = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // No padding: precomputed offsets are 0 and 4096.
  tt.func @andi_mask1_swizzled(%arg0: !ttg.memdesc<2x64x64xf16, #shared_sz, #smem, mutable>, %k: i32) {
    %c1_i32 = arith.constant 1 : i32
    %idx = arith.andi %k, %c1_i32 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-DAG: llvm.icmp "eq"
    // CHECK-DAG: llvm.mlir.constant(4096 : i32) : i32
    // CHECK: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_sz, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_sz, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 3. Two-element alphabet via arith.remui(x, 2)
// =============================================================================

// -----

// CHECK-LABEL: remui_mod2_padded
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Same expected offsets as `andi_mask1_padded` but driven by remui.
  tt.func @remui_mod2_padded(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c2_i32 = arith.constant 2 : i32
    %idx = arith.remui %k, %c2_i32 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.lshr
    // CHECK: llvm.mlir.constant(4224 : i32) : i32
    // CHECK: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 3b. Two-element alphabet via arith.remsi (signed remainder).
//     Hand-pipelined Gluon kernels use Python `%` which lowers to remsi.
// =============================================================================

// -----

// CHECK-LABEL: remsi_mod2_padded
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @remsi_mod2_padded(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c2_i32 = arith.constant 2 : i32
    %idx = arith.remsi %k, %c2_i32 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.lshr
    // CHECK-DAG: llvm.icmp "eq"
    // CHECK-DAG: llvm.mlir.constant(4224 : i32) : i32
    // CHECK: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 4. Four-element alphabet: arith.andi(x, 3)  (canonical form of `x % 4`)
// =============================================================================

// -----

// CHECK-LABEL: andi_mask3_swizzled_4buf
#shared_sz = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // 4-buffer alloc; alphabet = {0, 1, 2, 3}; offsets = {0, 4096, 8192, 12288}.
  // Expect three chained selects (idx==1, idx==2, idx==3) and no mul.
  tt.func @andi_mask3_swizzled_4buf(%arg0: !ttg.memdesc<4x64x64xf16, #shared_sz, #smem, mutable>, %k: i32) {
    %c3_i32 = arith.constant 3 : i32
    %idx = arith.andi %k, %c3_i32 : i32

    // CHECK-NOT: llvm.mul
    // The three non-zero buffer offsets (4096, 8192, 12288) appear inline
    // as constants feeding a 3-deep chain of selects.
    // CHECK: llvm.mlir.constant(4096 : i32)
    // CHECK: llvm.select
    // CHECK: llvm.mlir.constant(8192 : i32)
    // CHECK: llvm.select
    // CHECK: llvm.mlir.constant(12288 : i32)
    // CHECK: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<4x64x64xf16, #shared_sz, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_sz, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 5. Four-element alphabet via arith.remui(x, 4)
// =============================================================================

// -----

// CHECK-LABEL: remui_mod4_swizzled_4buf
#shared_sz = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @remui_mod4_swizzled_4buf(%arg0: !ttg.memdesc<4x64x64xf16, #shared_sz, #smem, mutable>, %k: i32) {
    %c4_i32 = arith.constant 4 : i32
    %idx = arith.remui %k, %c4_i32 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-COUNT-3: llvm.select
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<4x64x64xf16, #shared_sz, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_sz, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 6. Both cur and nxt projections in the same function: each memdesc_index
//    op produces its own select; total of two selects, no chain ops anywhere.
// =============================================================================

// -----

// CHECK-LABEL: cur_and_nxt_both_optimized
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func @cur_and_nxt_both_optimized(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c1_i32 = arith.constant 1 : i32
    %cur_idx = arith.andi %k, %c1_i32 : i32
    %k_plus_1 = arith.addi %k, %c1_i32 : i32
    %nxt_idx = arith.andi %k_plus_1, %c1_i32 : i32

    // CHECK-NOT: llvm.mul
    // CHECK-NOT: llvm.lshr
    // CHECK-NOT: llvm.shl
    // CHECK-COUNT-2: llvm.select

    %cur = ttg.memdesc_index %arg0[%cur_idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    %nxt = ttg.memdesc_index %arg0[%nxt_idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// =============================================================================
// 7. Negative tests: index shape not in the recognized alphabet must fall
//    back to the original `mul + applyPadding` chain.
// =============================================================================

// -----

// CHECK-LABEL: bare_index_falls_back
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Bare i32 argument: no defining op pattern is recognized.
  tt.func @bare_index_falls_back(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %idx: i32) {
    // CHECK-NOT: llvm.select
    // CHECK: llvm.mul
    // CHECK: llvm.lshr
    // CHECK: llvm.shl
    // CHECK: llvm.add
    // CHECK: llvm.getelementptr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: andi_mask_not_pow2_minus_1_falls_back
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // andi mask 5 is not (2^k - 1), so it does not generate a closed alphabet.
  tt.func @andi_mask_not_pow2_minus_1_falls_back(%arg0: !ttg.memdesc<8x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c5_i32 = arith.constant 5 : i32
    %idx = arith.andi %k, %c5_i32 : i32

    // CHECK-NOT: llvm.select
    // CHECK: llvm.mul
    // CHECK: llvm.lshr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<8x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: remui_modulus_exceeds_numbuffers_falls_back
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // remui modulus 4 > NUM_BUFFERS=2: out-of-range buffer indices possible.
  // Recognizer requires modulus == NUM_BUFFERS, so it falls back.
  tt.func @remui_modulus_exceeds_numbuffers_falls_back(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c4_i32 = arith.constant 4 : i32
    %idx = arith.remui %k, %c4_i32 : i32

    // CHECK-NOT: llvm.select
    // CHECK: llvm.mul
    // CHECK: llvm.lshr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: const_index_out_of_range_falls_back
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Constant index 5 with NUM_BUFFERS=2: out of range, recognizer refuses.
  tt.func @const_index_out_of_range_falls_back(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>) {
    %c5_i32 = arith.constant 5 : i32

    // CHECK-NOT: llvm.select
    // CHECK: llvm.mul

    %1 = ttg.memdesc_index %arg0[%c5_i32] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}

// -----

// CHECK-LABEL: muli_index_falls_back
#shared_pad = #ttg.padded_shared<[128:+4] {order = [1, 0], shape = [64, 64]}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "hip:gfx950", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 64 : i32} {
  // Multiplicative index pattern is not in the recognized set.
  tt.func @muli_index_falls_back(%arg0: !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable>, %k: i32) {
    %c2_i32 = arith.constant 2 : i32
    %idx = arith.muli %k, %c2_i32 : i32

    // CHECK-NOT: llvm.select
    // CHECK: llvm.mul
    // CHECK: llvm.lshr

    %1 = ttg.memdesc_index %arg0[%idx] : !ttg.memdesc<2x64x64xf16, #shared_pad, #smem, mutable> -> !ttg.memdesc<64x64xf16, #shared_pad, #smem, mutable>
    tt.return
  }
}
