#include "OffsetUniformitySplit.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>

namespace mlir::LLVM::AMD {
namespace {

// Env-gated debug output. Set TRITON_AMD_SOFFSET_SPLIT_DEBUG=1 to enable.
static bool debugEnabled() {
  static int cached = -1;
  if (cached == -1) {
    const char *v = std::getenv("TRITON_AMD_SOFFSET_SPLIT_DEBUG");
    cached = (v && v[0] != 0 && v[0] != '0') ? 1 : 0;
  }
  return cached == 1;
}

#define DBG() if (debugEnabled()) llvm::errs() << "[soffset-split] "

// One-line summary of a Value or its defining op.
static std::string describeValue(Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (Operation *def = v.getDefiningOp()) {
    os << def->getName().getStringRef();
  } else if (auto ba = dyn_cast<BlockArgument>(v)) {
    os << "BlockArg#" << ba.getArgNumber();
    Operation *parent = ba.getOwner()->getParentOp();
    if (parent)
      os << "(in " << parent->getName().getStringRef() << ")";
  } else {
    os << "<unknown>";
  }
  return s;
}

// Triton's tensor offsets reach the buffer-load conversion as scalar
// `llvm.extractvalue` ops sitting on top of a chain of `llvm.insertvalue`
// ops that pack the per-element offsets into a struct. Walk that chain to
// recover the actual offset SSA value at the requested position so the
// additive splitter and uniformity checker can see through it.
Value lookThroughExtractValue(Value v) {
  while (auto extract = v.getDefiningOp<LLVM::ExtractValueOp>()) {
    auto position = extract.getPosition();
    if (position.size() != 1)
      return v;
    int64_t targetIdx = position[0];
    Value cur = extract.getContainer();
    bool found = false;
    for (int steps = 0; steps < 4096 && cur; ++steps) {
      auto insert = cur.getDefiningOp<LLVM::InsertValueOp>();
      if (!insert)
        break;
      auto insertPos = insert.getPosition();
      if (insertPos.size() == 1 && insertPos[0] == targetIdx) {
        v = insert.getValue();
        found = true;
        break;
      }
      cur = insert.getContainer();
    }
    if (!found)
      return v;
  }
  return v;
}

class UniformityChecker {
public:
  bool isUniform(Value v) {
    auto it = cache.find(v);
    if (it != cache.end())
      return it->second;
    // Cycle break: optimistically assume uniform if we hit `v` mid-recursion
    // (e.g. `%k = phi [0, ...], [%k_next, ...]` where `%k_next = add(%k, c)`).
    // The result is finalized once the chain closes.
    if (!inFlight.insert(v).second)
      return true;
    bool result = compute(v);
    inFlight.erase(v);
    cache[v] = result;
    return result;
  }

  // Single-line reason why `v` is non-uniform, if known. Returns empty string
  // if the answer is "yes uniform" or if we don't have a reason cached.
  std::string whyNonUniform(Value v) {
    auto it = whyCache.find(v);
    if (it != whyCache.end())
      return it->second;
    return "";
  }

  // Chain-trace `v` to the deepest "root cause" non-uniform value the walker
  // would point at: a ThreadId* / LaneId, an unrecognised opcode, a
  // non-entry block argument, or an orphan. Returns a one-line summary.
  // Recurses for at most `maxDepth` hops to keep output bounded.
  std::string rootCause(Value v, unsigned maxDepth = 16) {
    Value cur = lookThroughExtractValue(v);
    for (unsigned d = 0; d < maxDepth; ++d) {
      auto it = whyCache.find(cur);
      if (it == whyCache.end())
        break;
      const std::string &reason = it->second;
      // If the reason references "operand #N of <op> is non-uniform:" we can
      // hop to that operand and keep climbing.
      auto pos = reason.find("operand #");
      if (pos == std::string::npos)
        return reason;
      Operation *def = cur.getDefiningOp();
      if (!def)
        return reason;
      // Walk operands and find the first non-uniform one (uses the cache).
      Value nextHop;
      for (Value op : def->getOperands()) {
        auto cIt = cache.find(op);
        if (cIt != cache.end() && cIt->second == false) {
          nextHop = op;
          break;
        }
      }
      if (!nextHop)
        return reason;
      cur = lookThroughExtractValue(nextHop);
    }
    auto it = whyCache.find(cur);
    if (it != whyCache.end())
      return it->second + "  [at depth " + std::to_string(maxDepth) + "]";
    return "<unknown>";
  }

private:
  void recordWhy(Value v, std::string reason) {
    if (debugEnabled() && whyCache.find(v) == whyCache.end())
      whyCache[v] = std::move(reason);
  }

  bool compute(Value v) {
    v = lookThroughExtractValue(v);

    if (auto blockArg = dyn_cast<BlockArgument>(v))
      return computeBlockArg(blockArg);

    Operation *def = v.getDefiningOp();
    if (!def) {
      recordWhy(v, "no defining op (orphan value)");
      return false;
    }

    if (isa<LLVM::ConstantOp, arith::ConstantOp, ROCDL::ReadfirstlaneOp,
            ROCDL::BlockIdXOp, ROCDL::BlockIdYOp, ROCDL::BlockIdZOp,
            ROCDL::WaveId>(def))
      return true;

    if (isa<ROCDL::ThreadIdXOp, ROCDL::ThreadIdYOp, ROCDL::ThreadIdZOp>(def)) {
      recordWhy(v, "ROCDL::ThreadId* (lane id)");
      return false;
    }

    if (isa<gpu::ThreadIdOp, gpu::LaneIdOp>(def)) {
      recordWhy(v, "gpu::ThreadId/LaneId");
      return false;
    }

    // rocdl.ds_bpermute(index, src) returns, for each receiving lane L,
    // src[ (index[L] >> 2) & 63 ]. If `index` is uniform across the wave,
    // every receiving lane reads the same source lane, so the result is
    // wave-uniform regardless of whether `src` is per-lane.
    if (auto bperm = dyn_cast<ROCDL::DsBpermuteOp>(def)) {
      if (isUniform(bperm.getIndex()))
        return true;
      recordWhy(v, "ds_bpermute with non-uniform index");
      return false;
    }

    // Pure arithmetic / cast ops are uniform iff every operand is uniform.
    if (isa<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::ShlOp, LLVM::LShrOp,
            LLVM::AShrOp, LLVM::AndOp, LLVM::OrOp, LLVM::XOrOp, LLVM::SExtOp,
            LLVM::ZExtOp, LLVM::TruncOp, LLVM::SelectOp, LLVM::ICmpOp,
            LLVM::URemOp, LLVM::SRemOp, LLVM::UDivOp, LLVM::SDivOp,
            LLVM::BitcastOp, LLVM::SMinOp, LLVM::SMaxOp, LLVM::UMinOp,
            LLVM::UMaxOp, LLVM::AbsOp, LLVM::PtrToIntOp, LLVM::IntToPtrOp,
            arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::SelectOp, arith::CmpIOp, arith::IndexCastOp,
            arith::IndexCastUIOp, arith::BitcastOp,
            arith::DivSIOp, arith::DivUIOp, arith::RemSIOp, arith::RemUIOp,
            arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(
            def)) {
      for (auto [idx, op] : llvm::enumerate(def->getOperands())) {
        if (!isUniform(op)) {
          recordWhy(v, ("operand #" + llvm::Twine(idx) + " of " +
                        def->getName().getStringRef() + " is non-uniform: " +
                        describeValue(op))
                           .str());
          return false;
        }
      }
      return true;
    }

    recordWhy(v, ("unrecognized op: " + def->getName().getStringRef()).str());
    return false;
  }

  bool computeBlockArg(BlockArgument blockArg) {
    Block *block = blockArg.getOwner();
    Operation *parent = block->getParentOp();
    if (block->isEntryBlock()) {
      // Entry block of a function: kernel arguments are uniform by
      // construction in Triton.
      if (isa<FunctionOpInterface>(parent))
        return true;
      // Entry block of a non-function region (e.g. an scf.for body): be
      // conservative; the producer may carry per-lane state.
      return false;
    }

    unsigned argIdx = blockArg.getArgNumber();
    for (Block *pred : block->getPredecessors()) {
      Operation *term = pred->getTerminator();
      if (!areIncomingValuesUniform(term, block, argIdx))
        return false;
    }
    return true;
  }

  bool areIncomingValuesUniform(Operation *term, Block *target,
                                unsigned argIdx) {
    if (auto br = dyn_cast<LLVM::BrOp>(term)) {
      auto operands = br.getDestOperands();
      if (argIdx < operands.size())
        return isUniform(operands[argIdx]);
      return false;
    }
    if (auto cb = dyn_cast<LLVM::CondBrOp>(term)) {
      bool sawTargetEdge = false;
      if (cb.getTrueDest() == target) {
        sawTargetEdge = true;
        auto operands = cb.getTrueDestOperands();
        if (argIdx >= operands.size() || !isUniform(operands[argIdx]))
          return false;
      }
      if (cb.getFalseDest() == target) {
        sawTargetEdge = true;
        auto operands = cb.getFalseDestOperands();
        if (argIdx >= operands.size() || !isUniform(operands[argIdx]))
          return false;
      }
      return sawTargetEdge;
    }
    return false;
  }

  llvm::DenseMap<Value, bool> cache;
  llvm::DenseSet<Value> inFlight;
  llvm::DenseMap<Value, std::string> whyCache;
};

bool isAddOrDisjointOr(Operation *op) {
  if (!op)
    return false;
  if (isa<LLVM::AddOp, arith::AddIOp>(op))
    return true;
  if (auto orOp = dyn_cast<LLVM::OrOp>(op))
    return orOp.getIsDisjoint();
  return false;
}

// Extract an integer constant from an LLVM or arith constant op.
std::optional<APInt> getIntConstant(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto cst = dyn_cast<LLVM::ConstantOp>(def)) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getValue();
  }
  if (auto cst = dyn_cast<arith::ConstantOp>(def)) {
    if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
      return attr.getValue();
  }
  return std::nullopt;
}

// A leaf with an accumulated multiplier chain. The leaf's contribution to
// the original offset is `value * constMul * (product of dynMuls)`. Both
// pieces start out at the multiplicative identity. No IR is created during
// the walk; materialization happens once the caller decides to commit.
struct LeafDescriptor {
  Value value;
  APInt constMul;
  SmallVector<Value, 2> dynMuls; // uniform Values to multiply (non-literal).
};

void collectAddTreeLeaves(Value v, APInt constMul,
                          ArrayRef<Value> dynMuls,
                          UniformityChecker &uniformity,
                          SmallVectorImpl<LeafDescriptor> &uniformLeaves,
                          SmallVectorImpl<LeafDescriptor> &perLaneLeaves) {
  v = lookThroughExtractValue(v);
  Operation *def = v.getDefiningOp();
  // Walk through every additive node we can recognize. Shared sub-trees get
  // re-summed in the partitioned output; downstream CSE/DCE folds duplicates.
  if (def && isAddOrDisjointOr(def)) {
    collectAddTreeLeaves(def->getOperand(0), constMul, dynMuls, uniformity,
                         uniformLeaves, perLaneLeaves);
    collectAddTreeLeaves(def->getOperand(1), constMul, dynMuls, uniformity,
                         uniformLeaves, perLaneLeaves);
    return;
  }
  // mul(x, y): if one operand is a literal int constant, accumulate it into
  // constMul. Otherwise, if one operand is wave-uniform, treat that as a
  // dynamic multiplier and descend into the other. Both cases let the outer
  // multiply distribute across the inner additive children at lift time.
  if (def && (isa<LLVM::MulOp, arith::MulIOp>(def))) {
    Value lhs = def->getOperand(0), rhs = def->getOperand(1);
    auto c0 = getIntConstant(lhs);
    auto c1 = getIntConstant(rhs);
    if (c1 && !c0) {
      APInt c = c1->sextOrTrunc(constMul.getBitWidth());
      DBG() << "  descend mul-by-const " << c << " into "
            << describeValue(lhs) << "\n";
      collectAddTreeLeaves(lhs, constMul * c, dynMuls, uniformity,
                           uniformLeaves, perLaneLeaves);
      return;
    }
    if (c0 && !c1) {
      APInt c = c0->sextOrTrunc(constMul.getBitWidth());
      DBG() << "  descend mul-by-const " << c << " into "
            << describeValue(rhs) << "\n";
      collectAddTreeLeaves(rhs, constMul * c, dynMuls, uniformity,
                           uniformLeaves, perLaneLeaves);
      return;
    }
    // Non-literal case: descend into the non-uniform side iff the other
    // side is wave-uniform. (uniform * uniform is handled as a leaf below;
    // per_lane * per_lane has no separable structure.)
    bool lhsU = uniformity.isUniform(lhs);
    bool rhsU = uniformity.isUniform(rhs);
    if (lhsU && !rhsU) {
      DBG() << "  descend mul-by-uniform " << describeValue(lhs) << " into "
            << describeValue(rhs) << "\n";
      SmallVector<Value> nextDyn(dynMuls.begin(), dynMuls.end());
      nextDyn.push_back(lhs);
      collectAddTreeLeaves(rhs, constMul, nextDyn, uniformity,
                           uniformLeaves, perLaneLeaves);
      return;
    }
    if (rhsU && !lhsU) {
      DBG() << "  descend mul-by-uniform " << describeValue(rhs) << " into "
            << describeValue(lhs) << "\n";
      SmallVector<Value> nextDyn(dynMuls.begin(), dynMuls.end());
      nextDyn.push_back(rhs);
      collectAddTreeLeaves(lhs, constMul, nextDyn, uniformity,
                           uniformLeaves, perLaneLeaves);
      return;
    }
  }
  // shl(x, k_const): equivalent to x * (1 << k_const).
  if (def && (isa<LLVM::ShlOp, arith::ShLIOp>(def))) {
    if (auto k = getIntConstant(def->getOperand(1))) {
      uint64_t shamt = k->getZExtValue();
      if (shamt < constMul.getBitWidth()) {
        APInt c = APInt(constMul.getBitWidth(), 1) << shamt;
        DBG() << "  descend shl-by-const " << shamt << " (mul " << c
              << ") into " << describeValue(def->getOperand(0)) << "\n";
        collectAddTreeLeaves(def->getOperand(0), constMul * c, dynMuls,
                             uniformity, uniformLeaves, perLaneLeaves);
        return;
      }
    }
  }
  LeafDescriptor desc{v, constMul, SmallVector<Value, 2>(dynMuls.begin(),
                                                          dynMuls.end())};
  if (uniformity.isUniform(v)) {
    DBG() << "  leaf UNIFORM    : " << describeValue(v) << "  constMul="
          << constMul << "  dynMul.size=" << dynMuls.size() << "\n";
    uniformLeaves.push_back(std::move(desc));
  } else {
    DBG() << "  leaf PER-LANE   : " << describeValue(v) << "  constMul="
          << constMul << "  dynMul.size=" << dynMuls.size()
          << "  root: " << uniformity.rootCause(v) << "\n";
    perLaneLeaves.push_back(std::move(desc));
  }
}

bool isLiteralZero(Value v) {
  if (auto c = getIntConstant(v))
    return c->isZero();
  return false;
}

// True iff the leaf contributes nothing: value is literal zero, or constant
// multiplier folded to zero. (Dynamic uniform multipliers are not assumed to
// be non-zero, but if they're statically zero `getIntConstant` would have
// caught them on the literal path.)
bool isZeroLeaf(const LeafDescriptor &leaf) {
  if (leaf.constMul.isZero())
    return true;
  return isLiteralZero(leaf.value);
}

// Materialize `leaf.value * constMul * dynMul[0] * dynMul[1] * ...`. If
// every factor is trivially the identity, returns the bare `value`.
Value materializeLeaf(const LeafDescriptor &leaf, RewriterBase &rewriter,
                      Location loc) {
  Value result = leaf.value;
  Type ty = result.getType();
  unsigned width = ty.getIntOrFloatBitWidth();
  if (leaf.constMul != 1) {
    Value c = LLVM::ConstantOp::create(
                  rewriter, loc, ty,
                  rewriter.getIntegerAttr(
                      ty, leaf.constMul.sextOrTrunc(width)))
                  .getResult();
    result = LLVM::MulOp::create(rewriter, loc, result, c).getResult();
  }
  for (Value m : leaf.dynMuls) {
    if (m.getType() != ty) {
      unsigned mw = m.getType().getIntOrFloatBitWidth();
      if (mw > width) {
        m = LLVM::TruncOp::create(rewriter, loc, ty, m).getResult();
      } else if (mw < width) {
        // Sign-extend dynamic multipliers (offsets are signed).
        m = LLVM::SExtOp::create(rewriter, loc, ty, m).getResult();
      }
    }
    result = LLVM::MulOp::create(rewriter, loc, result, m).getResult();
  }
  return result;
}

Value sumValues(ArrayRef<Value> vs, RewriterBase &rewriter, Location loc) {
  if (vs.empty())
    return Value();
  Value sum = vs.front();
  for (Value v : vs.drop_front())
    sum = LLVM::AddOp::create(rewriter, loc, sum, v).getResult();
  return sum;
}

} // namespace

std::pair<Value, Value> splitUniformAdditive(Value offset,
                                             RewriterBase &rewriter,
                                             Location loc) {
  if (debugEnabled()) {
    // Walk up the parent op chain to find the function name so each entry
    // is attributable to a kernel.
    StringRef fnName = "<unknown>";
    if (Operation *parent = rewriter.getInsertionBlock()
                                ? rewriter.getInsertionBlock()->getParentOp()
                                : nullptr) {
      Operation *cur = parent;
      while (cur && !isa<FunctionOpInterface>(cur))
        cur = cur->getParentOp();
      if (cur) {
        if (auto sym = cur->getAttrOfType<StringAttr>("sym_name"))
          fnName = sym.getValue();
      }
    }
    llvm::errs() << "[soffset-split] ===== fn=" << fnName
                 << "  offset_root=" << describeValue(offset) << "\n";
  }

  unsigned width = offset.getType().isIntOrIndex()
                       ? offset.getType().getIntOrFloatBitWidth()
                       : 32u;
  if (width == 0)
    width = 32u;

  UniformityChecker uniformity;
  SmallVector<LeafDescriptor> uniformLeaves;
  SmallVector<LeafDescriptor> perLaneLeaves;
  collectAddTreeLeaves(offset, APInt(width, 1), ArrayRef<Value>{}, uniformity,
                       uniformLeaves, perLaneLeaves);

  size_t uBefore = uniformLeaves.size();
  uniformLeaves.erase(
      std::remove_if(uniformLeaves.begin(), uniformLeaves.end(), isZeroLeaf),
      uniformLeaves.end());

  DBG() << "  summary uniform="
        << uniformLeaves.size() << " (was " << uBefore
        << " before zero-drop) perLane=" << perLaneLeaves.size();
  if (debugEnabled()) {
    llvm::errs() << (uniformLeaves.empty() ? "  -> NO SPLIT\n"
                                           : "  -> SPLIT\n");
  }

  if (uniformLeaves.empty())
    return {Value(), offset};

  // Materialize scaled uniform leaves and sum. No IR has been created up to
  // this point; everything from here forward goes through the rewriter.
  SmallVector<Value> uniformVals;
  uniformVals.reserve(uniformLeaves.size());
  for (const auto &leaf : uniformLeaves)
    uniformVals.push_back(materializeLeaf(leaf, rewriter, loc));
  Value uniform = sumValues(uniformVals, rewriter, loc);

  if (perLaneLeaves.empty()) {
    Value zero = LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                          rewriter.getI32IntegerAttr(0))
                     .getResult();
    return {uniform, zero};
  }
  SmallVector<Value> perLaneVals;
  perLaneVals.reserve(perLaneLeaves.size());
  for (const auto &leaf : perLaneLeaves)
    perLaneVals.push_back(materializeLeaf(leaf, rewriter, loc));
  Value perLane = sumValues(perLaneVals, rewriter, loc);
  return {uniform, perLane};
}

} // namespace mlir::LLVM::AMD
