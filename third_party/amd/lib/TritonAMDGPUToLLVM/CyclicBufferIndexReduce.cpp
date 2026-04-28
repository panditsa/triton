#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>
#include <optional>
#include <utility>

#define DEBUG_TYPE "tritonamdgpu-cyclic-buffer-index-reduce"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

// One canonical alternating-phi per (loop, mask). The phi starts at 0 and
// toggles to `mask` every iteration. The "next" value is a separate xor at
// the top of the loop header so it can be used inside the same iteration
// without violating SSA dominance. Both fields are LLVM IR pointers and
// stay valid for the entire pass even when the per-(loop, mask) cache
// DenseMap rehashes.
struct AlternatingPhi {
  PHINode *phi;
  Value *next; // xor(phi, mask)
};

// A binding records "this SSA value cycles through {0, mask} at iter k=0 if
// inverted is false, else {mask, 0}". `inverted == false` -> use `phi`;
// `inverted == true` -> use `next`. Stores raw IR pointers (not into the
// phi cache) so binding entries survive cache rehashing.
struct Binding {
  PHINode *phi;
  Value *next;
  bool inverted;
  uint64_t mask;
};

// Strip a chain of `add X, c` to recover the underlying value plus the
// accumulated constant offset.
Value *peelConstantAdds(Value *v, int64_t &accOffset) {
  accOffset = 0;
  Value *cur = v;
  while (true) {
    Value *inner = nullptr;
    ConstantInt *cInt = nullptr;
    if (match(cur, m_c_Add(m_Value(inner), m_ConstantInt(cInt)))) {
      accOffset += cInt->getSExtValue();
      cur = inner;
      continue;
    }
    break;
  }
  return cur;
}

// Identify a step-1 i32 add-recurrence with constant initial value defined
// in loop `L`. Returns the IV's initial value on success.
struct AffineIVInfo {
  Loop *loop;
  uint64_t initVal;
};

std::optional<AffineIVInfo> getStep1IV(Value *v, ScalarEvolution &SE,
                                       LoopInfo &LI) {
  if (!v->getType()->isIntegerTy(32))
    return std::nullopt;
  auto *inst = dyn_cast<Instruction>(v);
  if (!inst)
    return std::nullopt;
  Loop *L = LI.getLoopFor(inst->getParent());
  if (!L)
    return std::nullopt;
  if (!SE.isSCEVable(v->getType()))
    return std::nullopt;
  const SCEV *sv = SE.getSCEV(v);
  auto *ar = dyn_cast<SCEVAddRecExpr>(sv);
  if (!ar || ar->getLoop() != L || !ar->isAffine())
    return std::nullopt;
  auto *step = dyn_cast<SCEVConstant>(ar->getStepRecurrence(SE));
  if (!step || step->getValue()->getZExtValue() != 1)
    return std::nullopt;
  auto *startC = dyn_cast<SCEVConstant>(ar->getStart());
  if (!startC)
    return std::nullopt;
  return AffineIVInfo{L, startC->getValue()->getZExtValue()};
}

// Phase 1 pattern: `Y = and(shl(X, S), 1 << S)`. The mask isolates exactly
// the bit at the shift position, so `Y = (X & 1) << S`, which is a 2-state
// value depending on the parity of `X`. Returns (X, S) on success.
struct ShiftedSingleBitMatch {
  Value *x;
  uint64_t shiftAmt;
};

std::optional<ShiftedSingleBitMatch> matchShiftedSingleBit(Value *v) {
  if (!v->getType()->isIntegerTy(32))
    return std::nullopt;
  Value *x = nullptr;
  ConstantInt *shamtC = nullptr;
  ConstantInt *maskC = nullptr;
  if (!match(v, m_c_And(m_Shl(m_Value(x), m_ConstantInt(shamtC)),
                        m_ConstantInt(maskC))))
    return std::nullopt;
  uint64_t s = shamtC->getZExtValue();
  uint64_t m = maskC->getZExtValue();
  if (s >= 32 || m == 0)
    return std::nullopt;
  if (m != (uint64_t{1} << s))
    return std::nullopt;
  return ShiftedSingleBitMatch{x, s};
}

// Phase 2 pattern: `Y = or disjoint (lshr exact V, K), V` (in either operand
// order of the or). Both operands of the or refer to the SAME value V, with
// one wrapped in `lshr exact V, K`. Returns (V, K) on success.
struct LshrOrChainMatch {
  Value *v;
  uint64_t shrAmt;
};

std::optional<LshrOrChainMatch> matchLshrOrChain(Value *value) {
  if (!value->getType()->isIntegerTy(32))
    return std::nullopt;
  auto *binOp = dyn_cast<BinaryOperator>(value);
  if (!binOp || binOp->getOpcode() != Instruction::Or)
    return std::nullopt;
  // `or disjoint` is required: it is what allows V | (V >> K) to behave as
  // arithmetic addition when the bit ranges are disjoint, which is the
  // semantic guarantee we are folding on. The flag lives on the
  // PossiblyDisjointInst subclass of BinaryOperator.
  auto *disjointInst = dyn_cast<PossiblyDisjointInst>(binOp);
  if (!disjointInst || !disjointInst->isDisjoint())
    return std::nullopt;
  Value *opA = binOp->getOperand(0);
  Value *opB = binOp->getOperand(1);
  // Try both operand orderings: (lshr V, K) | V  and  V | (lshr V, K).
  for (int swap = 0; swap < 2; ++swap) {
    Value *lshrSide = swap ? opB : opA;
    Value *plainSide = swap ? opA : opB;
    Value *v = nullptr;
    ConstantInt *kC = nullptr;
    if (!match(lshrSide, m_LShr(m_Value(v), m_ConstantInt(kC))))
      continue;
    auto *lshrInst = dyn_cast<Instruction>(lshrSide);
    if (!lshrInst || !lshrInst->isExact())
      continue;
    if (v != plainSide)
      continue;
    uint64_t k = kC->getZExtValue();
    if (k == 0 || k >= 32)
      continue;
    return LshrOrChainMatch{v, k};
  }
  return std::nullopt;
}

struct CyclicBufferIndexReducePass : FunctionPass {
  CyclicBufferIndexReducePass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    if (F.empty())
      return false;

    AssumptionCache AC(F);
    TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
    TargetLibraryInfo TLI(TLII);
    DominatorTree DT(F);
    LoopInfo LI(DT);
    ScalarEvolution SE(F, TLI, AC, DT, LI);

    // Per-(loop, mask) cache of alt-phi pairs. Lives across phase 1 and the
    // iterative phase 2 so that derived phis can also be shared.
    DenseMap<std::pair<Loop *, uint64_t>, AlternatingPhi> phiCache;

    // SSA value -> binding into the alt-phi system. Used both to record what
    // we will rewrite at the end and as the lookup table for phase 2.
    DenseMap<Value *, Binding> bindings;

    auto getOrCreate = [&](Loop *L, uint64_t mask) -> AlternatingPhi {
      auto key = std::make_pair(L, mask);
      auto it = phiCache.find(key);
      if (it != phiCache.end())
        return it->second;

      BasicBlock *header = L->getHeader();
      BasicBlock *preheader = L->getLoopPreheader();
      BasicBlock *latch = L->getLoopLatch();
      if (!preheader || !latch)
        return AlternatingPhi{nullptr, nullptr};

      LLVMContext &ctx = header->getContext();
      IntegerType *i32Ty = Type::getInt32Ty(ctx);
      Constant *zero = ConstantInt::get(i32Ty, 0);
      Constant *maskC = ConstantInt::get(i32Ty, mask);

      // Insert the new phi at the very top of the header so it joins any
      // existing phi block. The xor goes immediately after all phis so its
      // value dominates every body use.
      PHINode *phi =
          PHINode::Create(i32Ty, /*NumReservedValues=*/2, "cyclic.alt",
                          header->begin());
      IRBuilder<> b(header, header->getFirstNonPHIIt());
      Value *next = b.CreateXor(phi, maskC, "cyclic.alt.next");
      phi->addIncoming(zero, preheader);
      phi->addIncoming(next, latch);

      AlternatingPhi result{phi, next};
      phiCache[key] = result;
      return result;
    };

    // Phase 1: scan for `(X << S) & (1 << S)` where X is a step-1 IV+const.
    unsigned phase1Hits = 0;
    for (BasicBlock &BB : F) {
      Loop *L = LI.getLoopFor(&BB);
      if (!L || !L->getLoopPreheader() || !L->getLoopLatch())
        continue;
      for (Instruction &inst : BB) {
        if (bindings.count(&inst))
          continue;
        auto matched = matchShiftedSingleBit(&inst);
        if (!matched)
          continue;
        int64_t addOffset = 0;
        Value *iv = peelConstantAdds(matched->x, addOffset);
        auto ivInfo = getStep1IV(iv, SE, LI);
        if (!ivInfo || ivInfo->loop != L)
          continue;

        uint64_t mask = uint64_t{1} << matched->shiftAmt;
        AlternatingPhi alt = getOrCreate(L, mask);
        if (!alt.phi)
          continue;

        // Canonical phi has init=0 and cycles {0, mask, 0, mask, ...}.
        // The match's value at iter 0 has parity (initVal + addOffset) & 1.
        bool inverted = (((uint64_t)((int64_t)ivInfo->initVal + addOffset)) &
                         1ULL) != 0;
        bindings[&inst] = Binding{alt.phi, alt.next, inverted, mask};
        ++phase1Hits;
      }
    }

    // Phase 2: iteratively scan for `or disjoint (lshr exact V, K), V` where
    // V is itself a known cyclic-2 value. The result cycles through
    // `{0, M | (M >> K)}` if V is canonical (or the inverted mirror if V is
    // inverted). Iterate until no more matches to handle deeper chains.
    unsigned phase2Hits = 0;
    bool changed = true;
    while (changed) {
      changed = false;
      for (BasicBlock &BB : F) {
        Loop *L = LI.getLoopFor(&BB);
        if (!L || !L->getLoopPreheader() || !L->getLoopLatch())
          continue;
        for (Instruction &inst : BB) {
          if (bindings.count(&inst))
            continue;
          auto matched = matchLshrOrChain(&inst);
          if (!matched)
            continue;
          auto vIt = bindings.find(matched->v);
          if (vIt == bindings.end())
            continue;
          // Copy out everything we need from V's binding BEFORE any
          // operation that can rehash `bindings` and invalidate `vIt`.
          Binding vBinding = vIt->second;
          // V's binding must be in the same loop as the chain.
          if (vBinding.phi->getParent() != L->getHeader())
            continue;
          uint64_t oldMask = vBinding.mask;
          uint64_t k = matched->shrAmt;
          if (k >= 32)
            continue;
          uint64_t shifted = oldMask >> k;
          // The fold is only valid when the two bit-ranges are disjoint;
          // the `or disjoint` flag asserts this for the IR, but we double
          // check on the masks to keep the new mask sound.
          if ((shifted & oldMask) != 0)
            continue;
          uint64_t newMask = oldMask | shifted;
          if (newMask == 0)
            continue;
          AlternatingPhi altNew = getOrCreate(L, newMask);
          if (!altNew.phi)
            continue;
          bindings[&inst] =
              Binding{altNew.phi, altNew.next, vBinding.inverted, newMask};
          ++phase2Hits;
          changed = true;
        }
      }
    }

    if (bindings.empty())
      return false;

    // Apply rewrites: replace each bound instruction with the appropriate
    // alt-phi value (canonical phi or its in-iter xor).
    SmallVector<Instruction *> toErase;
    for (auto &kv : bindings) {
      Instruction *inst = cast<Instruction>(kv.first);
      Value *replacement =
          kv.second.inverted ? kv.second.next : kv.second.phi;
      // Don't try to RAUW a value with itself (defensive; should not happen).
      if (replacement == inst)
        continue;
      inst->replaceAllUsesWith(replacement);
      toErase.push_back(inst);
    }
    for (Instruction *inst : toErase) {
      if (inst->use_empty())
        inst->eraseFromParent();
    }

    if (std::getenv("AMDGCN_CYCLIC_BUFFER_INDEX_REDUCE_VERBOSE") ||
        !bindings.empty()) {
      errs() << "[CyclicBufferIndexReduce] " << F.getName() << ": "
             << phase1Hits << " single-bit + " << phase2Hits
             << " lshr-or chain rewrite(s); " << phiCache.size()
             << " phi(s) inserted\n";
    }
    return true;
  }

  static char ID;
};

} // namespace

char CyclicBufferIndexReducePass::ID = 0;

namespace mlir::triton::AMD {
void runCyclicBufferIndexReducePass(llvm::Function &F) {
  CyclicBufferIndexReducePass pass;
  pass.runOnFunction(F);
  if (llvm::verifyFunction(F, &errs())) {
    errs()
        << "[CyclicBufferIndexReduce] WARNING: verifier failed after pass on "
        << F.getName() << "; downstream codegen may crash\n";
  }
}
} // namespace mlir::triton::AMD
