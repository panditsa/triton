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
#include <tuple>
#include <utility>

#define DEBUG_TYPE "tritonamdgpu-cyclic-buffer-index-reduce"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

// Result of matching `(X << S) & (1 << S)` where the mask isolates exactly
// the bit at the shift position. The matched value equals `(X & 1) << S`,
// which is a 2-state value depending on the parity of `X`.
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
  // Match `and(shl(X, S), M)` in either operand order of the and.
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

// One canonical alternating-phi per (loop, shiftAmt). The phi starts at 0
// and toggles to (1 << shiftAmt) every iteration. The "next" value is also
// materialized as a separate xor at the top of the loop header so it can be
// used inside the same iteration without violating SSA dominance.
struct AlternatingPhi {
  PHINode *phi;
  Value *next; // xor(phi, 1 << shiftAmt)
};

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

    using Key = std::pair<Loop *, uint64_t>;
    DenseMap<Key, AlternatingPhi> phiCache;

    // Each rewrite says: replace `inst` with either phi (if useNext is false)
    // or next (if useNext is true), for the (loop, shiftAmt) key.
    struct Rewrite {
      Instruction *inst;
      Key key;
      bool useNext;
    };
    SmallVector<Rewrite> rewrites;

    for (BasicBlock &BB : F) {
      Loop *L = LI.getLoopFor(&BB);
      if (!L)
        continue;
      // Need both preheader and latch to materialize the phi.
      if (!L->getLoopPreheader() || !L->getLoopLatch())
        continue;
      for (Instruction &inst : BB) {
        auto matched = matchShiftedSingleBit(&inst);
        if (!matched)
          continue;
        int64_t addOffset = 0;
        Value *iv = peelConstantAdds(matched->x, addOffset);
        auto ivInfo = getStep1IV(iv, SE, LI);
        if (!ivInfo || ivInfo->loop != L)
          continue;

        // Canonical phi has init=0 and cycles {0, mask, 0, mask, ...}.
        // We pick `phi` or `next` based on this match's parity at iter 0.
        uint64_t parityAtIter0 =
            ((uint64_t)((int64_t)ivInfo->initVal + addOffset)) & 1ULL;
        rewrites.push_back({&inst, {L, matched->shiftAmt}, parityAtIter0 != 0});
      }
    }

    if (rewrites.empty())
      return false;

    auto getOrCreate = [&](Loop *L, uint64_t shiftAmt) -> AlternatingPhi {
      Key key{L, shiftAmt};
      auto it = phiCache.find(key);
      if (it != phiCache.end())
        return it->second;

      BasicBlock *header = L->getHeader();
      BasicBlock *preheader = L->getLoopPreheader();
      BasicBlock *latch = L->getLoopLatch();
      LLVMContext &ctx = header->getContext();
      IntegerType *i32Ty = Type::getInt32Ty(ctx);
      uint64_t mask = uint64_t{1} << shiftAmt;
      Constant *zero = ConstantInt::get(i32Ty, 0);
      Constant *maskC = ConstantInt::get(i32Ty, mask);

      // PHI nodes must come at the top of the header. Insert at begin();
      // any existing phis stay grouped because the new phi joins them.
      PHINode *phi =
          PHINode::Create(i32Ty, /*NumReservedValues=*/2, "cyclic.alt",
                          header->begin());

      // The xor goes right after all phis so its value is available to all
      // body instructions without violating SSA dominance.
      IRBuilder<> b(header, header->getFirstNonPHIIt());
      Value *next = b.CreateXor(phi, maskC, "cyclic.alt.next");

      phi->addIncoming(zero, preheader);
      phi->addIncoming(next, latch);

      AlternatingPhi result{phi, next};
      phiCache[key] = result;
      return result;
    };

    unsigned numRewrites = 0;
    for (const Rewrite &r : rewrites) {
      AlternatingPhi alt = getOrCreate(r.key.first, r.key.second);
      Value *replacement = r.useNext ? alt.next : alt.phi;
      r.inst->replaceAllUsesWith(replacement);
      ++numRewrites;
    }
    for (const Rewrite &r : rewrites) {
      if (r.inst->use_empty())
        r.inst->eraseFromParent();
    }

    bool changed = numRewrites > 0;
    if (changed || std::getenv("AMDGCN_CYCLIC_BUFFER_INDEX_REDUCE_VERBOSE")) {
      errs() << "[CyclicBufferIndexReduce] " << F.getName() << ": "
             << numRewrites << " (X<<S)&(1<<S) -> alt-phi rewrite(s); "
             << phiCache.size() << " phi(s) inserted\n";
    }
    return changed;
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
