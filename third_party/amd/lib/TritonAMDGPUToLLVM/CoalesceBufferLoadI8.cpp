#include "TritonAMDGPUToLLVM/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Utils/Local.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#define DEBUG_TYPE "tritonamdgpu-coalesce-buffer-load-i8"

using namespace llvm;
using namespace llvm::PatternMatch;

namespace {

constexpr StringRef kBufferLoadI8 = "llvm.amdgcn.raw.ptr.buffer.load.i8";
constexpr StringRef kBufferLoadI32 = "llvm.amdgcn.raw.ptr.buffer.load.i32";

// Identify llvm.amdgcn.raw.ptr.buffer.load.i8 intrinsic calls.
bool isBufferLoadI8(const Instruction &inst) {
  auto *call = dyn_cast<CallInst>(&inst);
  if (!call || call->isInlineAsm())
    return false;
  Function *fn = call->getCalledFunction();
  return fn && fn->isIntrinsic() && fn->getName() == kBufferLoadI8;
}

bool isBufferLoadI32(Value *value) {
  auto *call = dyn_cast<CallInst>(value);
  if (!call || call->isInlineAsm())
    return false;
  Function *fn = call->getCalledFunction();
  return fn && fn->isIntrinsic() && fn->getName() == kBufferLoadI32;
}

bool isScaledMFMAIntrinsic(const CallBase *call) {
  if (!call || call->isInlineAsm())
    return false;
  Function *fn = call->getCalledFunction();
  return fn && fn->isIntrinsic() &&
         fn->getName().starts_with("llvm.amdgcn.mfma.scale.");
}

bool functionHasScaledMFMA(Function &F) {
  for (BasicBlock &BB : F)
    for (Instruction &inst : BB)
      if (isScaledMFMAIntrinsic(dyn_cast<CallBase>(&inst)))
        return true;
  return false;
}

bool isUsedByScaledMFMA(Value *value) {
  for (User *user : value->users())
    if (isScaledMFMAIntrinsic(dyn_cast<CallBase>(user)))
      return true;
  return false;
}

// Strip wrappers that constant-folding may have left around the offset arg
// of a `raw.ptr.buffer.load.i8` call.
Value *stripOffsetWrappers(Value *offset) {
  // Strip a `select(true, X, _)` wrapper if instcombine left it behind.
  if (auto *sel = dyn_cast<SelectInst>(offset))
    if (auto *cond = dyn_cast<ConstantInt>(sel->getCondition()))
      if (cond->isOne())
        return sel->getTrueValue();
  return offset;
}

// SCEV-backed constant-difference: returns the integer K such that
// SCEV(B) == SCEV(A) + K, if such a K exists.
std::optional<int64_t> scevConstDiff(ScalarEvolution &SE, Value *A, Value *B) {
  const SCEV *sA = SE.getSCEV(A);
  const SCEV *sB = SE.getSCEV(B);
  if (sA->getType() != sB->getType())
    return std::nullopt;
  const SCEV *diff = SE.getMinusSCEV(sB, sA);
  if (auto *c = dyn_cast<SCEVConstant>(diff))
    return c->getValue()->getSExtValue();
  return std::nullopt;
}

// True if the low log2(alignment) bits of `value` are provably zero.
bool isAlignedTo(Value *value, uint64_t alignment, const DataLayout &DL) {
  if (alignment <= 1)
    return true;
  KnownBits known = computeKnownBits(value, DL);
  uint64_t mask = alignment - 1;
  return (known.Zero.getZExtValue() & mask) == mask;
}

std::optional<int64_t> ptrSCEVConstDiff(ScalarEvolution &SE, Value *A,
                                        Value *B) {
  auto diff = scevConstDiff(SE, A, B);
  if (diff)
    return diff;
  return std::nullopt;
}

bool isI8OrV1I8Type(Type *ty) {
  if (ty->isIntegerTy(8))
    return true;
  auto *vecTy = dyn_cast<FixedVectorType>(ty);
  return vecTy && vecTy->getElementType()->isIntegerTy(8) &&
         vecTy->getNumElements() == 1;
}

bool isAddrSpace3Pointer(Value *ptr) {
  auto *ptrTy = dyn_cast<PointerType>(ptr->getType());
  return ptrTy && ptrTy->getAddressSpace() == 3;
}

Value *getStoredI8Scalar(StoreInst *store) {
  if (!store->isSimple())
    return nullptr;
  if (!isAddrSpace3Pointer(store->getPointerOperand()))
    return nullptr;

  Value *stored = store->getValueOperand();
  if (stored->getType()->isIntegerTy(8))
    return stored;

  auto *vecTy = dyn_cast<FixedVectorType>(stored->getType());
  if (!vecTy || !vecTy->getElementType()->isIntegerTy(8) ||
      vecTy->getNumElements() != 1)
    return nullptr;

  auto *insert = dyn_cast<InsertElementInst>(stored);
  if (!insert)
    return nullptr;
  auto *idx = dyn_cast<ConstantInt>(insert->getOperand(2));
  if (!idx || !idx->isZero())
    return nullptr;
  Value *lane = insert->getOperand(1);
  return lane->getType()->isIntegerTy(8) ? lane : nullptr;
}

bool isByteLoad(LoadInst *load) {
  return load && load->isSimple() && isI8OrV1I8Type(load->getType()) &&
         isAddrSpace3Pointer(load->getPointerOperand());
}

bool decomposeByteLoad(Value *value, LoadInst *&load) {
  if (auto *li = dyn_cast<LoadInst>(value)) {
    if (!isByteLoad(li))
      return false;
    load = li;
    return true;
  }

  auto *extract = dyn_cast<ExtractElementInst>(value);
  if (!extract)
    return false;
  auto *idx = dyn_cast<ConstantInt>(extract->getIndexOperand());
  if (!idx || !idx->isZero())
    return false;
  auto *li = dyn_cast<LoadInst>(extract->getVectorOperand());
  if (!isByteLoad(li))
    return false;
  load = li;
  return true;
}

std::optional<Value *> getVectorLane(Value *value, unsigned lane,
                                     unsigned depth = 0) {
  if (depth > 16)
    return std::nullopt;

  auto *vecTy = dyn_cast<FixedVectorType>(value->getType());
  if (!vecTy)
    return std::nullopt;
  if (lane >= vecTy->getNumElements())
    return std::nullopt;

  // A <1 x i8> value is already the byte carrier produced by Gluon LDS loads.
  if (vecTy->getElementType()->isIntegerTy(8) && vecTy->getNumElements() == 1 &&
      lane == 0)
    return value;

  if (auto *insert = dyn_cast<InsertElementInst>(value)) {
    auto *idx = dyn_cast<ConstantInt>(insert->getOperand(2));
    if (!idx)
      return std::nullopt;
    if (idx->getZExtValue() == lane)
      return insert->getOperand(1);
    return getVectorLane(insert->getOperand(0), lane, depth + 1);
  }

  if (auto *shuffle = dyn_cast<ShuffleVectorInst>(value)) {
    int mask = shuffle->getMaskValue(lane);
    if (mask < 0)
      return std::nullopt;
    Value *op0 = shuffle->getOperand(0);
    Value *op1 = shuffle->getOperand(1);
    auto *op0Ty = dyn_cast<FixedVectorType>(op0->getType());
    auto *op1Ty = dyn_cast<FixedVectorType>(op1->getType());
    if (!op0Ty || !op1Ty)
      return std::nullopt;
    unsigned op0Elems = op0Ty->getNumElements();
    if ((unsigned)mask < op0Elems)
      return getVectorLane(op0, (unsigned)mask, depth + 1);
    return getVectorLane(op1, (unsigned)mask - op0Elems, depth + 1);
  }

  return std::nullopt;
}

bool collectI8VectorBytes(Value *value, std::array<Value *, 4> &bytes) {
  auto *vecTy = dyn_cast<FixedVectorType>(value->getType());
  if (!vecTy || !vecTy->getElementType()->isIntegerTy(8) ||
      vecTy->getNumElements() != 4)
    return false;

  for (unsigned i = 0; i < 4; ++i) {
    auto lane = getVectorLane(value, i);
    if (!lane)
      return false;
    bytes[i] = *lane;
  }
  return true;
}

bool mayTouchLDSOrBarrier(const Instruction &inst) {
  auto touchesAS3Ptr = [](const Value *value) {
    auto *ptrTy = dyn_cast<PointerType>(value->getType());
    return ptrTy && ptrTy->getAddressSpace() == 3;
  };

  if (auto *load = dyn_cast<LoadInst>(&inst))
    return touchesAS3Ptr(load->getPointerOperand());
  if (auto *store = dyn_cast<StoreInst>(&inst))
    return touchesAS3Ptr(store->getPointerOperand());
  if (auto *rmw = dyn_cast<AtomicRMWInst>(&inst))
    return touchesAS3Ptr(rmw->getPointerOperand());
  if (auto *cmpxchg = dyn_cast<AtomicCmpXchgInst>(&inst))
    return touchesAS3Ptr(cmpxchg->getPointerOperand());

  auto *call = dyn_cast<CallBase>(&inst);
  if (!call)
    return false;

  Function *fn = call->getCalledFunction();
  if (fn && (fn->getName().starts_with("llvm.amdgcn.s.barrier") ||
             fn->getName().starts_with("llvm.amdgcn.wave.barrier") ||
             fn->getName().starts_with("llvm.amdgcn.s.waitcnt")))
    return true;

  if (!call->mayReadOrWriteMemory() && !call->mayHaveSideEffects())
    return false;
  for (Value *arg : call->args())
    if (touchesAS3Ptr(arg))
      return true;
  return false;
}

Instruction *earliestInBB(ArrayRef<Instruction *> insts) {
  Instruction *earliest = insts.front();
  for (Instruction *inst : insts) {
    if (inst == earliest)
      continue;
    if (inst->comesBefore(earliest))
      earliest = inst;
  }
  return earliest;
}

Instruction *latestInBB(ArrayRef<Instruction *> insts) {
  Instruction *latest = insts.front();
  for (Instruction *inst : insts) {
    if (inst == latest)
      continue;
    if (latest->comesBefore(inst))
      latest = inst;
  }
  return latest;
}

bool allInSameBB(ArrayRef<Instruction *> insts) {
  BasicBlock *BB = insts.front()->getParent();
  return llvm::all_of(insts,
                      [&](Instruction *inst) { return inst->getParent() == BB; });
}

bool hasInterveningLDSAlias(ArrayRef<Instruction *> clusterMembers) {
  if (!allInSameBB(clusterMembers))
    return true;

  SmallPtrSet<Instruction *, 8> clusterSet(clusterMembers.begin(),
                                           clusterMembers.end());
  Instruction *first = earliestInBB(clusterMembers);
  Instruction *last = latestInBB(clusterMembers);
  for (Instruction *inst = first->getNextNode(); inst && inst != last;
       inst = inst->getNextNode()) {
    if (clusterSet.contains(inst))
      continue;
    if (mayTouchLDSOrBarrier(*inst))
      return true;
  }
  return false;
}

// A run of N (= 2 or 4) byte-adjacent buffer_load_i8 calls.
struct Cluster {
  SmallVector<CallInst *, 4> calls; // ordered: byte 0, byte 1, [byte 2, byte 3]
  Value *anchorOffset;
  Value *rsrc;
  Value *sgprOff;
  Value *cacheMod;
};

// Find runs of 4 (preferred) or 2 (fallback) buffer_load_i8 calls in `BB`
// whose per-thread byte offsets are pairwise SCEV-constant deltas {0, 1,
// [2, 3]} from a common anchor and whose anchor offset is provably 4-byte
// (or 2-byte for pairs) aligned.
SmallVector<Cluster> findClustersInBB(BasicBlock &BB, ScalarEvolution &SE,
                                      const DataLayout &DL) {
  // Group calls by the `pre-offset` triple (rsrc, sgprOff, cacheMod). We
  // can't group by the offset SSA itself because each per-slot offset is a
  // distinct SSA value; SCEV is what proves they differ by a constant.
  using Key = std::tuple<Value *, Value *, Value *>;
  DenseMap<Key, SmallVector<CallInst *>> groups;

  for (Instruction &inst : BB) {
    if (!isBufferLoadI8(inst))
      continue;
    auto *call = cast<CallInst>(&inst);
    Value *rsrc = call->getArgOperand(0);
    Value *sgprOff = call->getArgOperand(2);
    Value *cacheMod = call->getArgOperand(3);
    groups[{rsrc, sgprOff, cacheMod}].push_back(call);
  }

  SmallVector<Cluster> clusters;
  for (auto &kv : groups) {
    auto &calls = kv.second;
    if (calls.size() < 2)
      continue;

    // Iterative anchor-rotation: each round picks the first remaining
    // call as the SCEV anchor and looks for a quartet first, then a
    // pair. Members of a found cluster are removed from `live`. If no
    // cluster is found for the current anchor, the anchor is dropped.
    SmallVector<CallInst *, 16> live(calls.begin(), calls.end());

    auto classifyFromAnchor = [&](CallInst *anchor) {
      Value *anchorOff = stripOffsetWrappers(anchor->getArgOperand(1));
      SmallVector<std::pair<int64_t, CallInst *>> classified;
      classified.push_back({0, anchor});
      for (CallInst *c : live) {
        if (c == anchor)
          continue;
        Value *off = stripOffsetWrappers(c->getArgOperand(1));
        auto delta = scevConstDiff(SE, anchorOff, off);
        if (delta)
          classified.push_back({*delta, c});
      }
      std::stable_sort(classified.begin(), classified.end(),
                       [](const std::pair<int64_t, CallInst *> &a,
                          const std::pair<int64_t, CallInst *> &b) {
                         return a.first < b.first;
                       });
      return std::make_pair(anchorOff, std::move(classified));
    };

    auto findClusterFor = [&](Value *anchorOff,
                              ArrayRef<std::pair<int64_t, CallInst *>>
                                  classified) -> std::optional<Cluster> {
      bool anchorAligned4 = isAlignedTo(anchorOff, 4, DL);
      bool anchorAligned2 = isAlignedTo(anchorOff, 2, DL);
      // Locate the anchor in `classified` (delta 0).
      size_t aIdx = 0;
      for (; aIdx < classified.size(); ++aIdx)
        if (classified[aIdx].first == 0)
          break;
      if (aIdx == classified.size())
        return std::nullopt;
      // Try quartet starting at aIdx (delta 0, 1, 2, 3).
      auto deltaAt = [&](size_t i) -> std::optional<int64_t> {
        if (i >= classified.size())
          return std::nullopt;
        return classified[i].first;
      };
      auto callAt = [&](size_t i) -> CallInst * {
        return classified[i].second;
      };
      if (anchorAligned4 && aIdx + 3 < classified.size() &&
          deltaAt(aIdx + 1) == 1 && deltaAt(aIdx + 2) == 2 &&
          deltaAt(aIdx + 3) == 3) {
        Cluster c;
        c.calls = {callAt(aIdx), callAt(aIdx + 1), callAt(aIdx + 2),
                   callAt(aIdx + 3)};
        c.anchorOffset = callAt(aIdx)->getArgOperand(1);
        return c;
      }
      if (anchorAligned2 && aIdx + 1 < classified.size() &&
          deltaAt(aIdx + 1) == 1) {
        Cluster c;
        c.calls = {callAt(aIdx), callAt(aIdx + 1)};
        c.anchorOffset = callAt(aIdx)->getArgOperand(1);
        return c;
      }
      return std::nullopt;
    };

    while (live.size() >= 2) {
      CallInst *anchor = live.front();
      auto [anchorOff, classified] = classifyFromAnchor(anchor);
      auto maybe = findClusterFor(anchorOff, classified);
      if (maybe) {
        Cluster c = *maybe;
        c.rsrc = std::get<0>(kv.first);
        c.sgprOff = std::get<1>(kv.first);
        c.cacheMod = std::get<2>(kv.first);
        // Remove all cluster members from live.
        for (CallInst *m : c.calls) {
          live.erase(std::remove(live.begin(), live.end(), m), live.end());
        }
        clusters.push_back(c);
      } else {
        // Drop this anchor; it never fused.
        live.erase(live.begin());
      }
    }
  }

  return clusters;
}

// Get-or-insert the `llvm.amdgcn.raw.ptr.buffer.load.<ty>` declaration.
// `ty` must be an integer type (i16 or i32 in our use).
Function *getOrInsertBufferLoad(Module *M, IntegerType *ty) {
  LLVMContext &ctx = M->getContext();
  Type *i32 = Type::getInt32Ty(ctx);
  Type *ptr8 = PointerType::get(ctx, /*addrSpace=*/8);
  FunctionType *ft = FunctionType::get(ty, {ptr8, i32, i32, i32}, false);
  std::string name =
      ty->getBitWidth() == 16 ? "llvm.amdgcn.raw.ptr.buffer.load.i16" :
      ty->getBitWidth() == 32 ? "llvm.amdgcn.raw.ptr.buffer.load.i32" :
      "llvm.amdgcn.raw.ptr.buffer.load.unknown";
  return cast<Function>(M->getOrInsertFunction(name, ft).getCallee());
}

// Find the call in `c` that comes first in its basic block.
CallInst *earliestInBB(const Cluster &c) {
  CallInst *earliest = c.calls[0];
  for (CallInst *call : c.calls) {
    if (earliest == call)
      continue;
    if (call->comesBefore(earliest))
      earliest = call;
  }
  return earliest;
}

bool rewriteCluster(const Cluster &c, IRBuilder<> &builder, Module *M) {
  // Insert at the earliest cluster member to dominate all original byte
  // uses (which all live downstream of their respective loads).
  CallInst *insertPt = earliestInBB(c);

  // Anchor offset (= smallest-byte address) must be defined at or before
  // the insertion point. If not, skip this cluster (would violate SSA).
  if (auto *anchorInst = dyn_cast<Instruction>(c.anchorOffset)) {
    if (anchorInst->getParent() == insertPt->getParent() &&
        !anchorInst->comesBefore(insertPt) && anchorInst != insertPt) {
      return false;
    }
  }
  builder.SetInsertPoint(insertPt);

  LLVMContext &ctx = M->getContext();
  IntegerType *i8Ty = Type::getInt8Ty(ctx);
  IntegerType *wideTy = c.calls.size() == 4 ? Type::getInt32Ty(ctx)
                                            : Type::getInt16Ty(ctx);

  Value *wideOff = c.anchorOffset;
  Function *bufLoadWide = getOrInsertBufferLoad(M, wideTy);
  CallInst *wide = builder.CreateCall(
      bufLoadWide, {c.rsrc, wideOff, c.sgprOff, c.cacheMod}, "coalesced.wide");
  wide->setOnlyReadsMemory();

  for (size_t i = 0; i < c.calls.size(); ++i) {
    Value *src = wide;
    if (i != 0)
      src = builder.CreateLShr(wide, ConstantInt::get(wideTy, i * 8),
                               "coalesced.byte.shr");
    Value *byteI8 = builder.CreateTrunc(src, i8Ty, "coalesced.byte");
    c.calls[i]->replaceAllUsesWith(byteI8);
  }
  return true;
}

struct LDSStoreCluster {
  std::array<StoreInst *, 4> stores;
  Value *basePtr;
  Value *dword;
};

struct LDSLoadCluster {
  BitCastInst *bitcast;
  std::array<LoadInst *, 4> loads;
  Value *basePtr;
};

// Decompose `byte` as a (dword, byte_index) pair where
// `byte == trunc i32 (lshr i32 dword, 8 * byte_index) to i8`.
// Returns false if the byte does not have that shape.
bool decomposeByteExtract(Value *byte, Value *&dword, unsigned &byteIdx) {
  auto *tr = dyn_cast<TruncInst>(byte);
  if (!tr || !tr->getDestTy()->isIntegerTy(8))
    return false;
  Value *src32 = tr->getOperand(0);
  if (!src32->getType()->isIntegerTy(32))
    return false;
  // i==0 case: bare trunc.
  if (auto *shr = dyn_cast<BinaryOperator>(src32)) {
    if (shr->getOpcode() == Instruction::LShr) {
      auto *shamt = dyn_cast<ConstantInt>(shr->getOperand(1));
      if (!shamt)
        return false;
      uint64_t s = shamt->getZExtValue();
      if (s % 8 != 0 || s > 24)
        return false;
      dword = shr->getOperand(0);
      byteIdx = s / 8;
      return true;
    }
  }
  dword = src32;
  byteIdx = 0;
  return true;
}

struct LDSByteStore {
  StoreInst *store;
  Value *dword;
  unsigned byteIdx;
};

SmallVector<LDSStoreCluster> findLDSStoreClustersInBB(BasicBlock &BB,
                                                      ScalarEvolution &SE,
                                                      const DataLayout &DL) {
  DenseMap<Value *, SmallVector<LDSByteStore, 8>> groups;

  for (Instruction &inst : BB) {
    auto *store = dyn_cast<StoreInst>(&inst);
    if (!store)
      continue;
    Value *byte = getStoredI8Scalar(store);
    if (!byte)
      continue;

    Value *dword = nullptr;
    unsigned byteIdx = 0;
    if (!decomposeByteExtract(byte, dword, byteIdx))
      continue;
    if (byteIdx >= 4 || !isBufferLoadI32(dword))
      continue;
    groups[dword].push_back({store, dword, byteIdx});
  }

  SmallVector<LDSStoreCluster> clusters;
  for (auto &kv : groups) {
    SmallVector<LDSByteStore, 8> live(kv.second.begin(), kv.second.end());
    while (live.size() >= 4) {
      auto anchorIt = llvm::find_if(live, [](const LDSByteStore &s) {
        return s.byteIdx == 0;
      });
      if (anchorIt == live.end())
        break;

      LDSStoreCluster cluster;
      cluster.stores = {nullptr, nullptr, nullptr, nullptr};
      cluster.dword = kv.first;
      cluster.stores[0] = anchorIt->store;
      cluster.basePtr = anchorIt->store->getPointerOperand();

      for (unsigned i = 1; i < 4; ++i) {
        auto memberIt = llvm::find_if(live, [&](const LDSByteStore &s) {
          if (s.byteIdx != i)
            return false;
          auto delta = ptrSCEVConstDiff(
              SE, cluster.basePtr, s.store->getPointerOperand());
          return delta && *delta == (int64_t)i;
        });
        if (memberIt != live.end())
          cluster.stores[i] = memberIt->store;
      }

      bool complete = llvm::all_of(cluster.stores,
                                   [](StoreInst *s) { return s != nullptr; });
      SmallVector<Instruction *, 4> members;
      if (complete) {
        for (StoreInst *store : cluster.stores)
          members.push_back(store);
      }

      if (complete && isAlignedTo(cluster.basePtr, 4, DL) &&
          !hasInterveningLDSAlias(members)) {
        clusters.push_back(cluster);
        for (StoreInst *store : cluster.stores) {
          live.erase(std::remove_if(live.begin(), live.end(),
                                    [&](const LDSByteStore &s) {
                                      return s.store == store;
                                    }),
                     live.end());
        }
        continue;
      }

      live.erase(anchorIt);
    }
  }

  return clusters;
}

bool rewriteLDSStoreCluster(const LDSStoreCluster &cluster,
                            IRBuilder<> &builder) {
  SmallVector<Instruction *, 4> members;
  for (StoreInst *store : cluster.stores)
    members.push_back(store);
  Instruction *insertPt = earliestInBB(members);

  if (auto *ptrInst = dyn_cast<Instruction>(cluster.basePtr)) {
    if (ptrInst->getParent() == insertPt->getParent() &&
        !ptrInst->comesBefore(insertPt) && ptrInst != insertPt)
      return false;
  }
  if (auto *dwordInst = dyn_cast<Instruction>(cluster.dword)) {
    if (dwordInst->getParent() == insertPt->getParent() &&
        !dwordInst->comesBefore(insertPt) && dwordInst != insertPt)
      return false;
  }

  builder.SetInsertPoint(insertPt);
  StoreInst *wideStore = builder.CreateStore(cluster.dword, cluster.basePtr);
  wideStore->setAlignment(Align(4));
  wideStore->setDebugLoc(insertPt->getDebugLoc());
  return true;
}

bool tryCollectLDSLoadCluster(BitCastInst *bc, ScalarEvolution &SE,
                              const DataLayout &DL,
                              LDSLoadCluster &cluster) {
  auto *dstTy = dyn_cast<IntegerType>(bc->getDestTy());
  if (!dstTy || dstTy->getBitWidth() != 32)
    return false;
  if (!isUsedByScaledMFMA(bc))
    return false;

  std::array<Value *, 4> bytes = {nullptr, nullptr, nullptr, nullptr};
  if (!collectI8VectorBytes(bc->getOperand(0), bytes))
    return false;

  cluster.bitcast = bc;
  cluster.loads = {nullptr, nullptr, nullptr, nullptr};
  for (unsigned i = 0; i < 4; ++i) {
    if (!decomposeByteLoad(bytes[i], cluster.loads[i]))
      return false;
  }

  cluster.basePtr = cluster.loads[0]->getPointerOperand();
  if (!isAlignedTo(cluster.basePtr, 4, DL))
    return false;

  for (unsigned i = 1; i < 4; ++i) {
    auto delta = ptrSCEVConstDiff(
        SE, cluster.basePtr, cluster.loads[i]->getPointerOperand());
    if (!delta || *delta != (int64_t)i)
      return false;
  }

  SmallVector<Instruction *, 4> members;
  for (LoadInst *load : cluster.loads)
    members.push_back(load);
  return !hasInterveningLDSAlias(members);
}

bool rewriteLDSLoadCluster(const LDSLoadCluster &cluster,
                           IRBuilder<> &builder) {
  SmallVector<Instruction *, 4> members;
  for (LoadInst *load : cluster.loads)
    members.push_back(load);
  Instruction *insertPt = earliestInBB(members);

  if (auto *ptrInst = dyn_cast<Instruction>(cluster.basePtr)) {
    if (ptrInst->getParent() == insertPt->getParent() &&
        !ptrInst->comesBefore(insertPt) && ptrInst != insertPt)
      return false;
  }

  builder.SetInsertPoint(insertPt);
  Type *i32Ty = Type::getInt32Ty(cluster.bitcast->getContext());
  LoadInst *wideLoad =
      builder.CreateLoad(i32Ty, cluster.basePtr, "coalesced.lds.dword");
  wideLoad->setAlignment(Align(4));
  wideLoad->setDebugLoc(insertPt->getDebugLoc());

  Value *oldVector = cluster.bitcast->getOperand(0);
  cluster.bitcast->replaceAllUsesWith(wideLoad);
  cluster.bitcast->eraseFromParent();
  if (auto *oldInst = dyn_cast<Instruction>(oldVector))
    RecursivelyDeleteTriviallyDeadInstructions(oldInst);
  return true;
}

bool tryFoldLDSVectorPhiScale(BitCastInst *bc, IRBuilder<> &builder) {
  auto *dstTy = dyn_cast<IntegerType>(bc->getDestTy());
  if (!dstTy || dstTy->getBitWidth() != 32)
    return false;
  if (!isUsedByScaledMFMA(bc))
    return false;

  auto *phi = dyn_cast<PHINode>(bc->getOperand(0));
  if (!phi)
    return false;

  auto *vecTy = dyn_cast<FixedVectorType>(phi->getType());
  if (!vecTy || !vecTy->getElementType()->isIntegerTy(8) ||
      vecTy->getNumElements() != 4)
    return false;

  SmallVector<LoadInst *, 4> incomingLoads;
  incomingLoads.reserve(phi->getNumIncomingValues());
  for (Value *incoming : phi->incoming_values()) {
    auto *load = dyn_cast<LoadInst>(incoming);
    if (!load || !load->isSimple())
      return false;
    if (load->getType() != phi->getType())
      return false;
    if (!isAddrSpace3Pointer(load->getPointerOperand()))
      return false;
    if (load->getAlign() < Align(4))
      return false;
    incomingLoads.push_back(load);
  }

  Type *i32Ty = Type::getInt32Ty(bc->getContext());
  builder.SetInsertPoint(phi->getParent(), phi->getParent()->getFirstNonPHIIt());
  PHINode *widePhi =
      builder.CreatePHI(i32Ty, phi->getNumIncomingValues(), "coalesced.lds.vec.phi");
  widePhi->setDebugLoc(phi->getDebugLoc());

  for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
    LoadInst *load = incomingLoads[i];
    IRBuilder<> loadBuilder(load->getNextNode());
    LoadInst *wideLoad = loadBuilder.CreateLoad(
        i32Ty, load->getPointerOperand(), "coalesced.lds.vec.dword");
    wideLoad->setAlignment(Align(4));
    wideLoad->setDebugLoc(load->getDebugLoc());
    widePhi->addIncoming(wideLoad, phi->getIncomingBlock(i));
  }

  bc->replaceAllUsesWith(widePhi);
  bc->eraseFromParent();
  RecursivelyDeleteTriviallyDeadInstructions(phi);
  return true;
}

bool tryFoldBytePhiRepack(BitCastInst *bc,
                          const std::array<Value *, 4> &bytes) {
  std::array<PHINode *, 4> phis = {nullptr, nullptr, nullptr, nullptr};
  for (int i = 0; i < 4; ++i) {
    phis[i] = dyn_cast<PHINode>(bytes[i]);
    if (!phis[i] || !phis[i]->getType()->isIntegerTy(8))
      return false;
    if (phis[i]->getNumIncomingValues() != phis[0]->getNumIncomingValues())
      return false;
    if (phis[i]->getParent() != phis[0]->getParent())
      return false;
  }

  unsigned numIncoming = phis[0]->getNumIncomingValues();
  SmallVector<Value *, 4> incomingDwords;
  incomingDwords.reserve(numIncoming);
  for (unsigned incomingIdx = 0; incomingIdx < numIncoming; ++incomingIdx) {
    BasicBlock *incomingBB = phis[0]->getIncomingBlock(incomingIdx);
    Value *dword = nullptr;
    for (int byteIdxExpected = 0; byteIdxExpected < 4; ++byteIdxExpected) {
      if (phis[byteIdxExpected]->getIncomingBlock(incomingIdx) != incomingBB)
        return false;

      Value *byteDword = nullptr;
      unsigned byteIdx = 0;
      if (!decomposeByteExtract(
              phis[byteIdxExpected]->getIncomingValue(incomingIdx), byteDword,
              byteIdx))
        return false;
      if (byteIdx != (unsigned)byteIdxExpected)
        return false;
      if (byteIdxExpected == 0) {
        dword = byteDword;
      } else if (dword != byteDword) {
        return false;
      }
    }
    incomingDwords.push_back(dword);
  }

  IRBuilder<> builder(phis[0]->getParent(),
                      phis[0]->getParent()->getFirstNonPHIIt());
  Type *i32Ty = Type::getInt32Ty(bc->getContext());
  PHINode *widePhi =
      builder.CreatePHI(i32Ty, numIncoming, "coalesced.scale.dword.phi");
  widePhi->setDebugLoc(phis[0]->getDebugLoc());
  for (unsigned i = 0; i < numIncoming; ++i)
    widePhi->addIncoming(incomingDwords[i], phis[0]->getIncomingBlock(i));

  bc->replaceAllUsesWith(widePhi);
  bc->eraseFromParent();
  for (PHINode *phi : phis)
    RecursivelyDeleteTriviallyDeadInstructions(phi);
  return true;
}

// Detect the pattern that AMD's convertScaledMFMA emits to build the packed
// i32 scale operand for the MFMA intrinsic:
//
//   %v0 = insertelement <4 x i8> undef, i8 %b0, i32 0
//   ...                                                  // 4 bytes total
//   %i32 = bitcast <4 x i8> %v3 to i32
//
// where each %bi = trunc i32 (lshr i32 %someDword, 8 * srcIdx_i) to i8.
//
// Two cases handled:
//  - Strict (1 dword): all 4 source dwords identical AND srcIdx_i == i.
//    Replace bitcast with %dword directly; zero ALU between load and MFMA.
//  - Permute (1 or 2 dwords): up to 2 distinct source dwords with arbitrary
//    byte indices per insertelement position. Replace with a single
//    `llvm.amdgcn.perm` call, which lowers to one `v_perm_b32`. This drops
//    the LLVM insertelement chain (and its lshr/and/or expansion) into one
//    intrinsic call.
bool tryFoldExtractRepack(BitCastInst *bc) {
  Value *src = bc->getOperand(0);
  auto *srcTy = dyn_cast<FixedVectorType>(src->getType());
  if (!srcTy)
    return false;
  if (!srcTy->getElementType()->isIntegerTy(8) ||
      srcTy->getNumElements() != 4)
    return false;
  auto *dstTy = dyn_cast<IntegerType>(bc->getDestTy());
  if (!dstTy || dstTy->getBitWidth() != 32)
    return false;
  bool dbg = std::getenv("AMDGCN_COALESCE_BUFFER_LOAD_I8_DEBUG_FOLD");
  std::array<Value *, 4> bytes = {nullptr, nullptr, nullptr, nullptr};
  if (!collectI8VectorBytes(src, bytes))
    return false;

  if (tryFoldBytePhiRepack(bc, bytes))
    return true;

  // Decompose each byte into (sourceDword, srcByteIndex).
  std::array<Value *, 4> srcDwords = {nullptr, nullptr, nullptr, nullptr};
  std::array<unsigned, 4> srcIdx = {0, 0, 0, 0};
  for (int i = 0; i < 4; ++i)
    if (!decomposeByteExtract(bytes[i], srcDwords[i], srcIdx[i]))
      return false;

  // Strict case: identical dword and canonical byte order.
  bool strict = true;
  for (int i = 0; i < 4; ++i) {
    if (srcDwords[i] != srcDwords[0] || srcIdx[i] != (unsigned)i) {
      strict = false;
      break;
    }
  }
  if (strict) {
    bc->replaceAllUsesWith(srcDwords[0]);
    bc->eraseFromParent();
    return true;
  }

  // Permute case: collect up to 2 distinct source dwords.
  Value *src0 = srcDwords[0];
  Value *src1 = nullptr;
  for (int i = 0; i < 4; ++i) {
    if (srcDwords[i] == src0)
      continue;
    if (src1 == nullptr) {
      src1 = srcDwords[i];
    } else if (srcDwords[i] != src1) {
      return false; // more than 2 distinct source dwords
    }
  }

  // Build the v_perm_b32 selector: one byte per output byte slot.
  // AMDGPU encoding: low 3 bits index a byte across {src1, src0}, where
  //   0..3 select src0 bytes 0..3
  //   4..7 select src1 bytes 0..3
  // (Note the AMD intrinsic takes args (src0, src1, sel) but the byte
  //  selector convention places src1 at indices 0..3 and src0 at 4..7.
  //  We pass operands so that our `src0` lives in selector range 4..7 and
  //  `src1` in 0..3, to match LLVM's lowering.)
  unsigned sel = 0;
  for (int i = 0; i < 4; ++i) {
    unsigned byteCode;
    if (srcDwords[i] == src1) {
      byteCode = srcIdx[i] & 0x3; // 0..3 picks src1 bytes
    } else {
      byteCode = (srcIdx[i] & 0x3) | 0x4; // 4..7 picks src0 bytes
    }
    sel |= (byteCode & 0xff) << (8 * i);
  }

  Module *M = bc->getModule();
  LLVMContext &ctx = M->getContext();
  Type *i32Ty = Type::getInt32Ty(ctx);
  FunctionType *fnTy =
      FunctionType::get(i32Ty, {i32Ty, i32Ty, i32Ty}, false);
  FunctionCallee permFn =
      M->getOrInsertFunction("llvm.amdgcn.perm", fnTy);
  IRBuilder<> builder(bc);
  Value *src1Val = src1 ? src1 : src0;
  Value *selVal = ConstantInt::get(i32Ty, sel);
  // amdgcn.perm signature is (s0, s1, sel) where selector values 0..3 pick
  // bytes of s1 and 4..7 pick bytes of s0. Our `src0` here lives in the
  // 4..7 selector region per the encoding above, so it goes first.
  Value *call = builder.CreateCall(permFn, {src0, src1Val, selVal},
                                   "coalesced.perm");
  (void)dbg;
  bc->replaceAllUsesWith(call);
  bc->eraseFromParent();
  return true;
}

struct CoalesceBufferLoadI8Pass : FunctionPass {
  CoalesceBufferLoadI8Pass() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override {
    bool changed = false;
    Module *M = F.getParent();
    const DataLayout &DL = M->getDataLayout();
    IRBuilder<> builder(F.getContext());

    // Build SCEV (and its analyses) standalone, mirroring how downstream
    // tools instantiate it without a PassManager.
    AssumptionCache AC(F);
    TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
    TargetLibraryInfo TLI(TLII);
    DominatorTree DT(F);
    LoopInfo LI(DT);
    ScalarEvolution SE(F, TLI, AC, DT, LI);

    unsigned totalI8Loads = 0;
    for (BasicBlock &BB : F)
      for (Instruction &inst : BB)
        if (isBufferLoadI8(inst))
          ++totalI8Loads;

    SmallVector<CallInst *> toErase;
    unsigned numQuartets = 0, numPairs = 0;
    for (BasicBlock &BB : F) {
      auto clusters = findClustersInBB(BB, SE, DL);
      for (auto &c : clusters) {
        if (!rewriteCluster(c, builder, M))
          continue;
        for (CallInst *call : c.calls)
          toErase.push_back(call);
        if (c.calls.size() == 4)
          ++numQuartets;
        else
          ++numPairs;
        changed = true;
      }
    }

    for (CallInst *c : toErase)
      c->eraseFromParent();

    unsigned numLDSStoreQuartets = 0;
    unsigned numLDSLoadQuartets = 0;
    unsigned numLDSVectorPhiFolds = 0;
    if (functionHasScaledMFMA(F)) {
      SmallVector<StoreInst *> storesToErase;
      SmallVector<Instruction *> storeValuesToDCE;
      for (BasicBlock &BB : F) {
        auto clusters = findLDSStoreClustersInBB(BB, SE, DL);
        for (const LDSStoreCluster &cluster : clusters) {
          for (StoreInst *store : cluster.stores)
            if (auto *valueInst =
                    dyn_cast<Instruction>(store->getValueOperand()))
              storeValuesToDCE.push_back(valueInst);
          if (!rewriteLDSStoreCluster(cluster, builder))
            continue;
          for (StoreInst *store : cluster.stores)
            storesToErase.push_back(store);
          ++numLDSStoreQuartets;
          changed = true;
        }
      }
      for (StoreInst *store : storesToErase)
        store->eraseFromParent();
      for (Instruction *value : storeValuesToDCE)
        RecursivelyDeleteTriviallyDeadInstructions(value);

      SmallVector<BitCastInst *> bitcastsToRewrite;
      for (BasicBlock &BB : F)
        for (Instruction &inst : BB)
          if (auto *bc = dyn_cast<BitCastInst>(&inst))
            bitcastsToRewrite.push_back(bc);
      for (BitCastInst *bc : bitcastsToRewrite) {
        LDSLoadCluster cluster;
        if (tryCollectLDSLoadCluster(bc, SE, DL, cluster)) {
          if (!rewriteLDSLoadCluster(cluster, builder))
            continue;
          ++numLDSLoadQuartets;
          changed = true;
          continue;
        }
        if (tryFoldLDSVectorPhiScale(bc, builder)) {
          ++numLDSVectorPhiFolds;
          changed = true;
        }
      }
    }

    // Second pass: fold the byte-extract + repack chain emitted by
    // convertScaledMFMA whenever all four bytes of a packed scale i32
    // come from the same dword we just loaded.
    unsigned numFolds = 0;
    SmallVector<BitCastInst *> bitcastsToTry;
    for (BasicBlock &BB : F)
      for (Instruction &inst : BB)
        if (auto *bc = dyn_cast<BitCastInst>(&inst))
          bitcastsToTry.push_back(bc);
    for (BitCastInst *bc : bitcastsToTry) {
      if (tryFoldExtractRepack(bc)) {
        ++numFolds;
        changed = true;
      }
    }

    if (changed || std::getenv("AMDGCN_COALESCE_BUFFER_LOAD_I8_VERBOSE")) {
      errs() << "[CoalesceBufferLoadI8] " << F.getName() << ": "
             << totalI8Loads << " i8 loads, coalesced " << numQuartets
             << " quartet(s) + " << numPairs
             << " pair(s); LDS coalesced " << numLDSStoreQuartets
             << " store quartet(s) + " << numLDSLoadQuartets
             << " load quartet(s) + " << numLDSVectorPhiFolds
             << " vector phi scale fold(s); folded " << numFolds
             << " extract+repack scale i32(s)\n";
    }
    return changed;
  }

  static char ID;
};

} // namespace

char CoalesceBufferLoadI8Pass::ID = 0;

namespace mlir::triton::AMD {
void runCoalesceBufferLoadI8Pass(llvm::Function &F) {
  CoalesceBufferLoadI8Pass pass;
  pass.runOnFunction(F);
  if (llvm::verifyFunction(F, &errs())) {
    errs() << "[CoalesceBufferLoadI8] WARNING: verifier failed after pass on "
           << F.getName() << "; downstream codegen may crash\n";
  }
}
} // namespace mlir::triton::AMD
