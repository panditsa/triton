#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/include/Analysis/AxisInfoExt.h"
#include "third_party/amd/include/Analysis/RangeAnalysis.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdlib>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonamdgpu-convert-buffer-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using ::mlir::LLVM::AMD::getVectorSize;

namespace ttg = mlir::triton::gpu;
namespace tt = mlir::triton;

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUCONVERTTOBUFFEROPS
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

// Return true iff the given value v is a tensor splatting from 1 (int).
// The usefulness of this func stems from the fact than if a buffer-op's mask
// operand is a all-1-tensor, it does not need to take this operand.
bool isSplatOneConstTensor(const Value v) {
  auto constantOp = v.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return false;

  if (auto denseAttr =
          dyn_cast<DenseIntElementsAttr>(constantOp.getValueAttr()))
    return denseAttr.isSplat() && denseAttr.getSplatValue<APInt>().isOne();

  return false;
}

// Returns true iff `solver` has a non-empty range for `v` whose minimum
// signed value is non-negative.
bool isProvenNonNegative(Value v, DataFlowSolver *solver) {
  const auto *lattice =
      solver->lookupState<dataflow::IntegerValueRangeLattice>(v);
  if (!lattice)
    return false;
  const mlir::IntegerValueRange &vr = lattice->getValue();
  if (vr.isUninitialized() || AMD::isEmptyInitializedRange(vr.getValue()))
    return false;
  return !vr.getValue().smin().isNegative();
}

Value sumIntegerValues(ArrayRef<Value> values, PatternRewriter &rewriter,
                       Location loc) {
  if (values.empty())
    return Value();
  Value sum = values.front();
  for (Value value : values.drop_front())
    sum = arith::AddIOp::create(rewriter, loc, sum, value);
  return sum;
}

Value createZeroTensorLike(Value tensor, PatternRewriter &rewriter,
                           Location loc) {
  auto tensorTy = cast<RankedTensorType>(tensor.getType());
  return arith::ConstantOp::create(rewriter, loc, tensorTy,
                                   rewriter.getZeroAttr(tensorTy));
}

std::optional<APInt> getIntegerConstant(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;

  Attribute attr = constantOp.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return intAttr.getValue();
  if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (denseAttr.isSplat())
      return denseAttr.getSplatValue<APInt>();
  }
  return std::nullopt;
}

std::optional<unsigned> getIntegerBitWidth(Type type) {
  Type scalarType = getElementTypeOrSelf(type);
  if (scalarType.isIndex())
    return 64;
  if (auto intType = dyn_cast<IntegerType>(scalarType))
    return intType.getWidth();
  return std::nullopt;
}

std::optional<APInt> getPossibleOneBits(Value value,
                                        DenseSet<Value> &active) {
  auto width = getIntegerBitWidth(value.getType());
  if (!width)
    return std::nullopt;

  APInt allOnes = APInt::getAllOnes(*width);
  if (!active.insert(value).second)
    return allOnes;
  auto finish = [&](std::optional<APInt> result) {
    active.erase(value);
    return result;
  };

  if (auto constant = getIntegerConstant(value))
    return finish(constant->sextOrTrunc(*width));

  if (auto makeRangeOp = value.getDefiningOp<tt::MakeRangeOp>()) {
    int64_t start = makeRangeOp.getStartAttr().getInt();
    int64_t end = makeRangeOp.getEndAttr().getInt();
    if (start < 0 || end <= start)
      return finish(allOnes);

    uint64_t maxValue = static_cast<uint64_t>(end - 1);
    unsigned bits = 0;
    while (maxValue) {
      ++bits;
      maxValue >>= 1;
    }
    return finish(APInt::getLowBitsSet(*width, std::min(bits, *width)));
  }

  if (auto splatOp = value.getDefiningOp<tt::SplatOp>()) {
    if (auto mask = getPossibleOneBits(splatOp.getSrc(), active))
      return finish(mask->zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto broadcastOp = value.getDefiningOp<tt::BroadcastOp>()) {
    if (auto mask = getPossibleOneBits(broadcastOp.getSrc(), active))
      return finish(mask->zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto expandOp = value.getDefiningOp<tt::ExpandDimsOp>()) {
    if (auto mask = getPossibleOneBits(expandOp.getSrc(), active))
      return finish(mask->zextOrTrunc(*width));
    return finish(std::nullopt);
  }

  if (auto truncOp = value.getDefiningOp<arith::TruncIOp>()) {
    if (auto mask = getPossibleOneBits(truncOp.getIn(), active))
      return finish(mask->trunc(*width));
    return finish(std::nullopt);
  }
  if (auto extOp = value.getDefiningOp<arith::ExtUIOp>()) {
    if (auto mask = getPossibleOneBits(extOp.getIn(), active))
      return finish(mask->zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto extOp = value.getDefiningOp<arith::ExtSIOp>()) {
    auto srcWidth = getIntegerBitWidth(extOp.getIn().getType());
    auto mask = getPossibleOneBits(extOp.getIn(), active);
    if (!srcWidth || !mask)
      return finish(std::nullopt);
    APInt result = mask->zextOrTrunc(*width);
    if (*srcWidth < *width && (*mask)[*srcWidth - 1])
      result.setBitsFrom(*srcWidth);
    return finish(result);
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
    if (auto mask = getPossibleOneBits(castOp.getIn(), active))
      return finish(mask->zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastUIOp>()) {
    if (auto mask = getPossibleOneBits(castOp.getIn(), active))
      return finish(mask->zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto bitcastOp = value.getDefiningOp<arith::BitcastOp>()) {
    auto srcWidth = getIntegerBitWidth(bitcastOp.getIn().getType());
    if (!srcWidth || *srcWidth != *width)
      return finish(allOnes);
    if (auto mask = getPossibleOneBits(bitcastOp.getIn(), active))
      return finish(*mask);
    return finish(std::nullopt);
  }

  if (auto andOp = value.getDefiningOp<arith::AndIOp>()) {
    auto lhs = getPossibleOneBits(andOp.getLhs(), active);
    auto rhs = getPossibleOneBits(andOp.getRhs(), active);
    if (lhs && rhs)
      return finish((*lhs & *rhs).zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto orOp = value.getDefiningOp<arith::OrIOp>()) {
    auto lhs = getPossibleOneBits(orOp.getLhs(), active);
    auto rhs = getPossibleOneBits(orOp.getRhs(), active);
    if (lhs && rhs)
      return finish((*lhs | *rhs).zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto xorOp = value.getDefiningOp<arith::XOrIOp>()) {
    auto lhs = getPossibleOneBits(xorOp.getLhs(), active);
    auto rhs = getPossibleOneBits(xorOp.getRhs(), active);
    if (lhs && rhs)
      return finish((*lhs | *rhs).zextOrTrunc(*width));
    return finish(std::nullopt);
  }
  if (auto shlOp = value.getDefiningOp<arith::ShLIOp>()) {
    auto shift = getIntegerConstant(shlOp.getRhs());
    auto lhs = getPossibleOneBits(shlOp.getLhs(), active);
    if (shift && lhs && !shift->isNegative()) {
      uint64_t shiftAmount = shift->getZExtValue();
      if (shiftAmount < *width)
        return finish((*lhs << shiftAmount).zextOrTrunc(*width));
    }
    return finish(allOnes);
  }
  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    Value lhsValue = mulOp.getLhs();
    Value rhsValue = mulOp.getRhs();
    auto lhsConst = getIntegerConstant(lhsValue);
    auto rhsConst = getIntegerConstant(rhsValue);
    Value shiftedValue;
    APInt factor(*width, 0);
    if (lhsConst && !rhsConst) {
      shiftedValue = rhsValue;
      factor = lhsConst->zextOrTrunc(*width);
    } else if (rhsConst && !lhsConst) {
      shiftedValue = lhsValue;
      factor = rhsConst->zextOrTrunc(*width);
    }
    if (shiftedValue && factor.isZero())
      return finish(APInt::getZero(*width));
    if (shiftedValue && factor.isPowerOf2()) {
      for (unsigned shiftAmount = 0; shiftAmount < *width; ++shiftAmount) {
        if (!factor[shiftAmount])
          continue;
        auto mask = getPossibleOneBits(shiftedValue, active);
        if (mask)
          return finish((*mask << shiftAmount).zextOrTrunc(*width));
        return finish(std::nullopt);
      }
    }
    return finish(allOnes);
  }

  return finish(allOnes);
}

bool isKnownDisjointOr(arith::OrIOp orOp) {
  DenseSet<Value> active;
  auto lhs = getPossibleOneBits(orOp.getLhs(), active);
  active.clear();
  auto rhs = getPossibleOneBits(orOp.getRhs(), active);
  if (!lhs || !rhs)
    return false;
  return (*lhs & *rhs).isZero();
}

class HighLevelUniformityChecker {
public:
  bool isUniform(Value value) {
    auto it = cache.find(value);
    if (it != cache.end())
      return it->second;
    if (!inFlight.insert(value).second)
      return true;
    bool result = compute(value);
    inFlight.erase(value);
    cache[value] = result;
    return result;
  }

private:
  bool compute(Value value) {
    if (auto blockArg = dyn_cast<BlockArgument>(value))
      return computeBlockArg(blockArg);

    Operation *def = value.getDefiningOp();
    if (!def)
      return false;

    if (isa<arith::ConstantOp, tt::GetProgramIdOp, tt::GetNumProgramsOp>(def))
      return true;

    if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      auto result = cast<OpResult>(value);
      unsigned resultIdx = result.getResultNumber();
      if (resultIdx >= forOp.getInitArgs().size())
        return false;
      if (!isUniform(forOp.getInitArgs()[resultIdx]))
        return false;
      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yieldOp || resultIdx >= yieldOp.getResults().size())
        return false;
      return isUniform(yieldOp.getResults()[resultIdx]);
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      if (!isUniform(ifOp.getCondition()))
        return false;
      auto result = cast<OpResult>(value);
      unsigned resultIdx = result.getResultNumber();
      auto thenYield = dyn_cast<scf::YieldOp>(
          ifOp.getThenRegion().front().getTerminator());
      if (!thenYield || resultIdx >= thenYield.getResults().size())
        return false;
      if (!isUniform(thenYield.getResults()[resultIdx]))
        return false;
      if (ifOp.getElseRegion().empty())
        return false;
      auto elseYield = dyn_cast<scf::YieldOp>(
          ifOp.getElseRegion().front().getTerminator());
      if (!elseYield || resultIdx >= elseYield.getResults().size())
        return false;
      return isUniform(elseYield.getResults()[resultIdx]);
    }

    if (isa<tt::SplatOp, tt::BroadcastOp, tt::ExpandDimsOp>(def)) {
      for (Value operand : def->getOperands())
        if (!isUniform(operand))
          return false;
      return true;
    }

    if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
            arith::ShRSIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
            arith::SelectOp, arith::CmpIOp, arith::IndexCastOp,
            arith::IndexCastUIOp, arith::BitcastOp, arith::DivSIOp,
            arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::MinSIOp,
            arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp>(def)) {
      for (Value operand : def->getOperands())
        if (!isUniform(operand))
          return false;
      return true;
    }

    return false;
  }

  bool computeBlockArg(BlockArgument blockArg) {
    Block *block = blockArg.getOwner();
    Operation *parent = block->getParentOp();
    if (block->isEntryBlock() && isa<FunctionOpInterface>(parent))
      return true;

    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (blockArg.getArgNumber() == 0)
        return isUniform(forOp.getLowerBound()) &&
               isUniform(forOp.getUpperBound()) && isUniform(forOp.getStep());
      unsigned iterIdx = blockArg.getArgNumber() - 1;
      if (iterIdx >= forOp.getInitArgs().size())
        return false;
      if (!isUniform(forOp.getInitArgs()[iterIdx]))
        return false;
      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yieldOp || iterIdx >= yieldOp.getResults().size())
        return false;
      return isUniform(yieldOp.getResults()[iterIdx]);
    }

    return false;
  }

  DenseMap<Value, bool> cache;
  DenseSet<Value> inFlight;
};

struct TensorProjection {
  enum class Kind { Broadcast, ExpandDims };

  Kind kind;
  RankedTensorType resultType;
  uint32_t axis = 0;
};

struct HighLevelOffsetLeaf {
  Value value;
  APInt constMul;
  SmallVector<Value, 2> dynMuls;
  SmallVector<TensorProjection, 4> projections;
};

SmallVector<TensorProjection, 4>
appendProjection(ArrayRef<TensorProjection> projections,
                 TensorProjection projection) {
  SmallVector<TensorProjection, 4> result(projections.begin(),
                                          projections.end());
  result.push_back(projection);
  return result;
}

bool canMaterializeUniformScalar(Value value,
                                 HighLevelUniformityChecker &uniformity,
                                 DenseSet<Value> &active) {
  if (!uniformity.isUniform(value))
    return false;
  if (!isa<RankedTensorType>(value.getType()))
    return true;
  if (!active.insert(value).second)
    return true;
  auto finish = [&](bool result) {
    active.erase(value);
    return result;
  };

  if (auto constantOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(constantOp.getValue()))
      return finish(denseAttr.isSplat());
    return finish(false);
  }
  if (auto splatOp = value.getDefiningOp<tt::SplatOp>())
    return finish(canMaterializeUniformScalar(splatOp.getSrc(), uniformity,
                                              active));
  if (auto broadcastOp = value.getDefiningOp<tt::BroadcastOp>())
    return finish(canMaterializeUniformScalar(broadcastOp.getSrc(), uniformity,
                                              active));
  if (auto expandOp = value.getDefiningOp<tt::ExpandDimsOp>())
    return finish(canMaterializeUniformScalar(expandOp.getSrc(), uniformity,
                                              active));

  Operation *def = value.getDefiningOp();
  if (!def)
    return finish(false);

  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ShLIOp,
          arith::ShRSIOp, arith::ShRUIOp, arith::AndIOp, arith::OrIOp,
          arith::XOrIOp, arith::DivSIOp, arith::DivUIOp, arith::RemSIOp,
          arith::RemUIOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
          arith::MaxUIOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
          arith::IndexCastOp, arith::IndexCastUIOp, arith::BitcastOp,
          arith::CmpIOp, arith::SelectOp>(def)) {
    for (Value operand : def->getOperands())
      if (!canMaterializeUniformScalar(operand, uniformity, active))
        return finish(false);
    return finish(true);
  }

  return finish(false);
}

bool canMaterializeUniformScalar(Value value,
                                 HighLevelUniformityChecker &uniformity) {
  DenseSet<Value> active;
  return canMaterializeUniformScalar(value, uniformity, active);
}

void collectHighLevelOffsetLeaves(
    Value offset, APInt constMul, ArrayRef<Value> dynMuls,
    ArrayRef<TensorProjection> projections,
    SmallVectorImpl<HighLevelOffsetLeaf> &uniformLeaves,
    SmallVectorImpl<HighLevelOffsetLeaf> &perLaneLeaves,
    HighLevelUniformityChecker &uniformity) {
  if (auto addOp = offset.getDefiningOp<arith::AddIOp>()) {
    collectHighLevelOffsetLeaves(addOp.getLhs(), constMul, dynMuls,
                                 projections, uniformLeaves, perLaneLeaves,
                                 uniformity);
    collectHighLevelOffsetLeaves(addOp.getRhs(), constMul, dynMuls,
                                 projections, uniformLeaves, perLaneLeaves,
                                 uniformity);
    return;
  }

  if (auto orOp = offset.getDefiningOp<arith::OrIOp>()) {
    if (isKnownDisjointOr(orOp)) {
      collectHighLevelOffsetLeaves(orOp.getLhs(), constMul, dynMuls,
                                   projections, uniformLeaves, perLaneLeaves,
                                   uniformity);
      collectHighLevelOffsetLeaves(orOp.getRhs(), constMul, dynMuls,
                                   projections, uniformLeaves, perLaneLeaves,
                                   uniformity);
      return;
    }
  }

  if (auto broadcastOp = offset.getDefiningOp<tt::BroadcastOp>()) {
    auto nextProjections = appendProjection(
        projections,
        {TensorProjection::Kind::Broadcast, broadcastOp.getType(), 0});
    collectHighLevelOffsetLeaves(broadcastOp.getSrc(), constMul, dynMuls,
                                 nextProjections, uniformLeaves, perLaneLeaves,
                                 uniformity);
    return;
  }

  if (auto expandOp = offset.getDefiningOp<tt::ExpandDimsOp>()) {
    auto nextProjections = appendProjection(
        projections,
        {TensorProjection::Kind::ExpandDims, expandOp.getType(),
         expandOp.getAxis()});
    collectHighLevelOffsetLeaves(expandOp.getSrc(), constMul, dynMuls,
                                 nextProjections, uniformLeaves, perLaneLeaves,
                                 uniformity);
    return;
  }

  if (auto mulOp = offset.getDefiningOp<arith::MulIOp>()) {
    Value lhs = mulOp.getLhs();
    Value rhs = mulOp.getRhs();
    auto lhsConst = getIntegerConstant(lhs);
    auto rhsConst = getIntegerConstant(rhs);
    if (rhsConst && !lhsConst) {
      collectHighLevelOffsetLeaves(lhs,
                                   constMul *
                                       rhsConst->sextOrTrunc(
                                           constMul.getBitWidth()),
                                   dynMuls, projections, uniformLeaves,
                                   perLaneLeaves, uniformity);
      return;
    }
    if (lhsConst && !rhsConst) {
      collectHighLevelOffsetLeaves(rhs,
                                   constMul *
                                       lhsConst->sextOrTrunc(
                                           constMul.getBitWidth()),
                                   dynMuls, projections, uniformLeaves,
                                   perLaneLeaves, uniformity);
      return;
    }

    bool lhsUniform = canMaterializeUniformScalar(lhs, uniformity);
    bool rhsUniform = canMaterializeUniformScalar(rhs, uniformity);
    if (lhsUniform && rhsUniform) {
      uniformLeaves.push_back({offset, constMul,
                               SmallVector<Value, 2>(dynMuls.begin(),
                                                     dynMuls.end()),
                               SmallVector<TensorProjection, 4>(
                                   projections.begin(), projections.end())});
      return;
    }
    if (lhsUniform && !rhsUniform) {
      SmallVector<Value, 2> nextDyn(dynMuls.begin(), dynMuls.end());
      nextDyn.push_back(lhs);
      collectHighLevelOffsetLeaves(rhs, constMul, nextDyn, projections,
                                   uniformLeaves, perLaneLeaves, uniformity);
      return;
    }
    if (rhsUniform && !lhsUniform) {
      SmallVector<Value, 2> nextDyn(dynMuls.begin(), dynMuls.end());
      nextDyn.push_back(rhs);
      collectHighLevelOffsetLeaves(lhs, constMul, nextDyn, projections,
                                   uniformLeaves, perLaneLeaves, uniformity);
      return;
    }
  }

  if (auto shlOp = offset.getDefiningOp<arith::ShLIOp>()) {
    if (auto shift = getIntegerConstant(shlOp.getRhs())) {
      uint64_t shiftAmount = shift->getZExtValue();
      if (shiftAmount < constMul.getBitWidth()) {
        APInt scale = APInt(constMul.getBitWidth(), 1) << shiftAmount;
        collectHighLevelOffsetLeaves(shlOp.getLhs(), constMul * scale,
                                     dynMuls, projections, uniformLeaves,
                                     perLaneLeaves, uniformity);
        return;
      }
    }
  }

  if (canMaterializeUniformScalar(offset, uniformity)) {
    uniformLeaves.push_back({offset, constMul,
                             SmallVector<Value, 2>(dynMuls.begin(),
                                                   dynMuls.end()),
                             SmallVector<TensorProjection, 4>(
                                 projections.begin(), projections.end())});
    return;
  }

  perLaneLeaves.push_back({offset, constMul,
                           SmallVector<Value, 2>(dynMuls.begin(),
                                                 dynMuls.end()),
                           SmallVector<TensorProjection, 4>(
                               projections.begin(), projections.end())});
}

Value castScalarInteger(Value value, Type targetType,
                        PatternRewriter &rewriter, Location loc) {
  if (value.getType() == targetType)
    return value;
  if (value.getType().isIndex() || targetType.isIndex())
    return arith::IndexCastOp::create(rewriter, loc, targetType, value);

  auto srcType = cast<IntegerType>(value.getType());
  auto dstType = cast<IntegerType>(targetType);
  if (srcType.getWidth() > dstType.getWidth())
    return arith::TruncIOp::create(rewriter, loc, targetType, value);
  return arith::ExtSIOp::create(rewriter, loc, targetType, value);
}

Value createIntegerConstantLike(Type resultType, APInt value,
                                PatternRewriter &rewriter, Location loc) {
  Type scalarType = getElementTypeOrSelf(resultType);
  unsigned width = scalarType.isIndex()
                       ? value.getBitWidth()
                       : cast<IntegerType>(scalarType).getWidth();
  Value scalar = arith::ConstantOp::create(
      rewriter, loc, scalarType,
      rewriter.getIntegerAttr(scalarType, value.sextOrTrunc(width)));
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType))
    return tt::SplatOp::create(rewriter, loc, tensorType, scalar);
  return scalar;
}

Value materializeUniformScalar(Value value, PatternRewriter &rewriter,
                               Location loc);

Value materializeUniformOperand(Value value, Type targetType,
                                PatternRewriter &rewriter, Location loc) {
  Value scalar = materializeUniformScalar(value, rewriter, loc);
  if (!scalar)
    return Value();
  return castScalarInteger(scalar, targetType, rewriter, loc);
}

Value materializeUniformScalar(Value value, PatternRewriter &rewriter,
                               Location loc) {
  if (!isa<RankedTensorType>(value.getType()))
    return value;

  Type scalarType = getElementTypeOrSelf(value.getType());
  if (auto constant = getIntegerConstant(value))
    return createIntegerConstantLike(scalarType, *constant, rewriter, loc);
  if (auto splatOp = value.getDefiningOp<tt::SplatOp>())
    return materializeUniformScalar(splatOp.getSrc(), rewriter, loc);
  if (auto broadcastOp = value.getDefiningOp<tt::BroadcastOp>())
    return materializeUniformScalar(broadcastOp.getSrc(), rewriter, loc);
  if (auto expandOp = value.getDefiningOp<tt::ExpandDimsOp>())
    return materializeUniformScalar(expandOp.getSrc(), rewriter, loc);

  if (auto addOp = value.getDefiningOp<arith::AddIOp>()) {
    Value lhs = materializeUniformOperand(addOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(addOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::AddIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto subOp = value.getDefiningOp<arith::SubIOp>()) {
    Value lhs = materializeUniformOperand(subOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(subOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::SubIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto mulOp = value.getDefiningOp<arith::MulIOp>()) {
    Value lhs = materializeUniformOperand(mulOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(mulOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::MulIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto shlOp = value.getDefiningOp<arith::ShLIOp>()) {
    Value lhs = materializeUniformOperand(shlOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(shlOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::ShLIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto shrOp = value.getDefiningOp<arith::ShRSIOp>()) {
    Value lhs = materializeUniformOperand(shrOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(shrOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::ShRSIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto shrOp = value.getDefiningOp<arith::ShRUIOp>()) {
    Value lhs = materializeUniformOperand(shrOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(shrOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::ShRUIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto andOp = value.getDefiningOp<arith::AndIOp>()) {
    Value lhs = materializeUniformOperand(andOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(andOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::AndIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto orOp = value.getDefiningOp<arith::OrIOp>()) {
    Value lhs = materializeUniformOperand(orOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(orOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::OrIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto xorOp = value.getDefiningOp<arith::XOrIOp>()) {
    Value lhs = materializeUniformOperand(xorOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(xorOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::XOrIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto divOp = value.getDefiningOp<arith::DivSIOp>()) {
    Value lhs = materializeUniformOperand(divOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(divOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::DivSIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto divOp = value.getDefiningOp<arith::DivUIOp>()) {
    Value lhs = materializeUniformOperand(divOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(divOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::DivUIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto remOp = value.getDefiningOp<arith::RemSIOp>()) {
    Value lhs = materializeUniformOperand(remOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(remOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::RemSIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto remOp = value.getDefiningOp<arith::RemUIOp>()) {
    Value lhs = materializeUniformOperand(remOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(remOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::RemUIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto minOp = value.getDefiningOp<arith::MinSIOp>()) {
    Value lhs = materializeUniformOperand(minOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(minOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::MinSIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto minOp = value.getDefiningOp<arith::MinUIOp>()) {
    Value lhs = materializeUniformOperand(minOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(minOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::MinUIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto maxOp = value.getDefiningOp<arith::MaxSIOp>()) {
    Value lhs = materializeUniformOperand(maxOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(maxOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::MaxSIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto maxOp = value.getDefiningOp<arith::MaxUIOp>()) {
    Value lhs = materializeUniformOperand(maxOp.getLhs(), scalarType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(maxOp.getRhs(), scalarType, rewriter,
                                          loc);
    return arith::MaxUIOp::create(rewriter, loc, lhs, rhs);
  }
  if (auto cmpOp = value.getDefiningOp<arith::CmpIOp>()) {
    Type operandType = getElementTypeOrSelf(cmpOp.getLhs().getType());
    Value lhs = materializeUniformOperand(cmpOp.getLhs(), operandType, rewriter,
                                          loc);
    Value rhs = materializeUniformOperand(cmpOp.getRhs(), operandType, rewriter,
                                          loc);
    return arith::CmpIOp::create(rewriter, loc, cmpOp.getPredicate(), lhs,
                                 rhs);
  }
  if (auto selectOp = value.getDefiningOp<arith::SelectOp>()) {
    Value cond = materializeUniformOperand(selectOp.getCondition(),
                                           rewriter.getI1Type(), rewriter, loc);
    Value trueValue = materializeUniformOperand(
        selectOp.getTrueValue(), scalarType, rewriter, loc);
    Value falseValue = materializeUniformOperand(
        selectOp.getFalseValue(), scalarType, rewriter, loc);
    return arith::SelectOp::create(rewriter, loc, cond, trueValue, falseValue);
  }

  if (auto castOp = value.getDefiningOp<arith::ExtSIOp>()) {
    Value input = materializeUniformScalar(castOp.getIn(), rewriter, loc);
    if (input.getType() == scalarType)
      return input;
    return arith::ExtSIOp::create(rewriter, loc, scalarType, input);
  }
  if (auto castOp = value.getDefiningOp<arith::ExtUIOp>()) {
    Value input = materializeUniformScalar(castOp.getIn(), rewriter, loc);
    if (input.getType() == scalarType)
      return input;
    return arith::ExtUIOp::create(rewriter, loc, scalarType, input);
  }
  if (auto castOp = value.getDefiningOp<arith::TruncIOp>()) {
    Value input = materializeUniformScalar(castOp.getIn(), rewriter, loc);
    if (input.getType() == scalarType)
      return input;
    return arith::TruncIOp::create(rewriter, loc, scalarType, input);
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>()) {
    Value input = materializeUniformScalar(castOp.getIn(), rewriter, loc);
    if (input.getType() == scalarType)
      return input;
    return arith::IndexCastOp::create(rewriter, loc, scalarType, input);
  }
  if (auto castOp = value.getDefiningOp<arith::IndexCastUIOp>()) {
    Value input = materializeUniformScalar(castOp.getIn(), rewriter, loc);
    if (input.getType() == scalarType)
      return input;
    return arith::IndexCastUIOp::create(rewriter, loc, scalarType, input);
  }
  if (auto castOp = value.getDefiningOp<arith::BitcastOp>()) {
    Value input = materializeUniformScalar(castOp.getIn(), rewriter, loc);
    if (input.getType() == scalarType)
      return input;
    return arith::BitcastOp::create(rewriter, loc, scalarType, input);
  }

  return Value();
}

Value materializeScalarFactorLike(Value factor, Type resultType,
                                  PatternRewriter &rewriter, Location loc) {
  Type scalarType = getElementTypeOrSelf(resultType);
  factor = materializeUniformScalar(factor, rewriter, loc);
  factor = castScalarInteger(factor, scalarType, rewriter, loc);
  if (auto tensorType = dyn_cast<RankedTensorType>(resultType))
    return tt::SplatOp::create(rewriter, loc, tensorType, factor);
  return factor;
}

Value materializeUniformHighLevelLeaf(const HighLevelOffsetLeaf &leaf,
                                      PatternRewriter &rewriter,
                                      Location loc) {
  Value result = materializeUniformScalar(leaf.value, rewriter, loc);
  if (leaf.constMul != 1) {
    Value factor = createIntegerConstantLike(result.getType(), leaf.constMul,
                                             rewriter, loc);
    result = arith::MulIOp::create(rewriter, loc, result, factor);
  }
  for (Value dynMul : leaf.dynMuls) {
    Value factor = materializeUniformScalar(dynMul, rewriter, loc);
    factor = castScalarInteger(factor, result.getType(), rewriter, loc);
    result = arith::MulIOp::create(rewriter, loc, result, factor);
  }
  return result;
}

Value materializeHighLevelLeaf(const HighLevelOffsetLeaf &leaf,
                               PatternRewriter &rewriter, Location loc,
                               bool applyProjections) {
  Value result = leaf.value;
  if (leaf.constMul != 1) {
    Value factor =
        createIntegerConstantLike(result.getType(), leaf.constMul, rewriter,
                                  loc);
    result = arith::MulIOp::create(rewriter, loc, result, factor);
  }
  for (Value dynMul : leaf.dynMuls) {
    Value factor =
        materializeScalarFactorLike(dynMul, result.getType(), rewriter, loc);
    result = arith::MulIOp::create(rewriter, loc, result, factor);
  }

  if (!applyProjections)
    return result;

  for (auto it = leaf.projections.rbegin(); it != leaf.projections.rend();
       ++it) {
    if (it->kind == TensorProjection::Kind::Broadcast) {
      result =
          tt::BroadcastOp::create(rewriter, loc, it->resultType, result);
      continue;
    }
    result = tt::ExpandDimsOp::create(rewriter, loc, result, it->axis);
  }
  return result;
}

bool isZeroLeaf(const HighLevelOffsetLeaf &leaf) {
  if (leaf.constMul.isZero())
    return true;
  if (auto constant = getIntegerConstant(leaf.value))
    return constant->isZero();
  return false;
}

bool isKnownNonNegative(Value value, DataFlowSolver *solver,
                        DenseSet<Value> &active) {
  if (isProvenNonNegative(value, solver))
    return true;
  if (!active.insert(value).second)
    return true;

  if (auto constant = getIntegerConstant(value))
    return !constant->isNegative();

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parent = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (blockArg.getArgNumber() == 0)
        return isKnownNonNegative(forOp.getLowerBound(), solver, active) &&
               isKnownNonNegative(forOp.getStep(), solver, active);
      unsigned iterIdx = blockArg.getArgNumber() - 1;
      if (iterIdx >= forOp.getInitArgs().size())
        return false;
      if (!isKnownNonNegative(forOp.getInitArgs()[iterIdx], solver, active))
        return false;
      auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yieldOp || iterIdx >= yieldOp.getResults().size())
        return false;
      return isKnownNonNegative(yieldOp.getResults()[iterIdx], solver, active);
    }
  }

  if (isa_and_nonnull<tt::GetProgramIdOp, tt::GetNumProgramsOp>(
          value.getDefiningOp()))
    return true;
  if (auto makeRangeOp = value.getDefiningOp<tt::MakeRangeOp>())
    return makeRangeOp.getStartAttr().getInt() >= 0;
  if (auto splatOp = value.getDefiningOp<tt::SplatOp>())
    return isKnownNonNegative(splatOp.getSrc(), solver, active);
  if (auto broadcastOp = value.getDefiningOp<tt::BroadcastOp>())
    return isKnownNonNegative(broadcastOp.getSrc(), solver, active);
  if (auto expandOp = value.getDefiningOp<tt::ExpandDimsOp>())
    return isKnownNonNegative(expandOp.getSrc(), solver, active);
  if (auto truncOp = value.getDefiningOp<arith::TruncIOp>())
    return isKnownNonNegative(truncOp.getIn(), solver, active);
  if (value.getDefiningOp<arith::ExtUIOp>())
    return true;
  if (auto extOp = value.getDefiningOp<arith::ExtSIOp>())
    return isKnownNonNegative(extOp.getIn(), solver, active);
  if (auto castOp = value.getDefiningOp<arith::IndexCastOp>())
    return isKnownNonNegative(castOp.getIn(), solver, active);
  if (value.getDefiningOp<arith::IndexCastUIOp>())
    return true;
  if (auto forOp = value.getDefiningOp<scf::ForOp>()) {
    auto result = cast<OpResult>(value);
    unsigned resultIdx = result.getResultNumber();
    if (resultIdx >= forOp.getInitArgs().size())
      return false;
    if (!isKnownNonNegative(forOp.getInitArgs()[resultIdx], solver, active))
      return false;
    auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yieldOp || resultIdx >= yieldOp.getResults().size())
      return false;
    return isKnownNonNegative(yieldOp.getResults()[resultIdx], solver, active);
  }
  if (auto ifOp = value.getDefiningOp<scf::IfOp>()) {
    auto result = cast<OpResult>(value);
    unsigned resultIdx = result.getResultNumber();
    auto thenYield = dyn_cast<scf::YieldOp>(
        ifOp.getThenRegion().front().getTerminator());
    if (!thenYield || resultIdx >= thenYield.getResults().size() ||
        !isKnownNonNegative(thenYield.getResults()[resultIdx], solver, active))
      return false;
    if (ifOp.getElseRegion().empty())
      return false;
    auto elseYield = dyn_cast<scf::YieldOp>(
        ifOp.getElseRegion().front().getTerminator());
    return elseYield && resultIdx < elseYield.getResults().size() &&
           isKnownNonNegative(elseYield.getResults()[resultIdx], solver,
                              active);
  }
  if (auto addOp = value.getDefiningOp<arith::AddIOp>())
    return isKnownNonNegative(addOp.getLhs(), solver, active) &&
           isKnownNonNegative(addOp.getRhs(), solver, active);
  if (auto mulOp = value.getDefiningOp<arith::MulIOp>())
    return isKnownNonNegative(mulOp.getLhs(), solver, active) &&
           isKnownNonNegative(mulOp.getRhs(), solver, active);
  if (auto shlOp = value.getDefiningOp<arith::ShLIOp>())
    return isKnownNonNegative(shlOp.getLhs(), solver, active) &&
           isKnownNonNegative(shlOp.getRhs(), solver, active);
  if (auto andOp = value.getDefiningOp<arith::AndIOp>())
    return isKnownNonNegative(andOp.getLhs(), solver, active) &&
           isKnownNonNegative(andOp.getRhs(), solver, active);

  return false;
}

bool isLeafProvenNonNegative(const HighLevelOffsetLeaf &leaf,
                             DataFlowSolver *solver) {
  if (leaf.constMul.isZero())
    return true;
  if (leaf.constMul.isNegative())
    return false;

  DenseSet<Value> active;
  if (!isKnownNonNegative(leaf.value, solver, active))
    return false;
  for (Value dynMul : leaf.dynMuls) {
    active.clear();
    if (!isKnownNonNegative(dynMul, solver, active))
      return false;
  }
  return true;
}

bool isRelaxableRem(Value value, DataFlowSolver *solver) {
  if (value.getDefiningOp<arith::RemUIOp>())
    return true;
  if (auto remOp = value.getDefiningOp<arith::RemSIOp>()) {
    DenseSet<Value> active;
    return isKnownNonNegative(remOp.getLhs(), solver, active);
  }
  return false;
}

bool isRelaxableNonNegativeLeaf(const HighLevelOffsetLeaf &leaf,
                                DataFlowSolver *solver,
                                bool allowLoopCarried) {
  if (leaf.constMul.isZero())
    return true;
  if (leaf.constMul.isNegative())
    return false;

  Value value = leaf.value;
  if (auto makeRangeOp = value.getDefiningOp<tt::MakeRangeOp>())
    return makeRangeOp.getStartAttr().getInt() >= 0;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return allowLoopCarried && isa<scf::ForOp>(blockArg.getOwner()->getParentOp());

  return isRelaxableRem(value, solver);
}

bool isRelaxablePackedByteLeaf(const HighLevelOffsetLeaf &leaf,
                               DataFlowSolver *solver) {
  if (leaf.constMul.isZero())
    return true;
  if (leaf.constMul.isNegative())
    return false;

  Value value = leaf.value;
  if (auto makeRangeOp = value.getDefiningOp<tt::MakeRangeOp>())
    return makeRangeOp.getStartAttr().getInt() >= 0;
  return isRelaxableRem(value, solver);
}

enum class RelaxedNonNegativeMode { None, NonByte, NonByteLoadToLocal, PackedByte };

bool hasNonByteElement(Type type) {
  Type elementType = type;
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type))
    elementType = memDescType.getElementType();
  else
    elementType = getElementTypeOrSelf(type);
  return elementType.isIntOrFloat() && elementType.getIntOrFloatBitWidth() > 8;
}

bool hasByteElement(Type type) {
  Type elementType = type;
  if (auto memDescType = dyn_cast<ttg::MemDescType>(type))
    elementType = memDescType.getElementType();
  else
    elementType = getElementTypeOrSelf(type);
  return elementType.isIntOrFloat() && elementType.getIntOrFloatBitWidth() == 8;
}

// Env-gated diagnostic for the MLIR-level walker. Set
// TRITON_AMD_BUFFER_OPS_DEBUG=1 to enable.
static bool mlirWalkerDebug() {
  static int cached = -1;
  if (cached == -1) {
    const char *v = std::getenv("TRITON_AMD_BUFFER_OPS_DEBUG");
    cached = (v && v[0] != 0 && v[0] != '0') ? 1 : 0;
  }
  return cached == 1;
}

// Experimental escape hatch for coverage experiments. The default remains the
// strict per-leaf non-negativity proof because AMD buffer bounds checks do not
// include soffset.
static bool relaxedNonNegativeSplit() {
  static int cached = -1;
  if (cached == -1) {
    const char *v = std::getenv("TRITON_AMD_BUFFER_OPS_RELAXED_NONNEG");
    cached = (v && v[0] != 0 && v[0] != '0') ? 1 : 0;
  }
  return cached == 1;
}

static std::string describeMlirValue(Value v) {
  std::string s;
  llvm::raw_string_ostream os(s);
  if (Operation *def = v.getDefiningOp()) {
    os << def->getName().getStringRef();
  } else if (auto ba = dyn_cast<BlockArgument>(v)) {
    os << "BlockArg#" << ba.getArgNumber();
    if (auto p = ba.getOwner()->getParentOp())
      os << "(in " << p->getName().getStringRef() << ")";
  } else {
    os << "<unknown>";
  }
  return s;
}

std::pair<Value, Value> splitHighLevelBufferOffset(Value offset,
                                                   PatternRewriter &rewriter,
                                                   DataFlowSolver *solver,
                                                   RelaxedNonNegativeMode relaxedMode) {
  Type scalarOffsetType = getElementTypeOrSelf(offset.getType());
  unsigned width = 32;
  if (auto intType = dyn_cast<IntegerType>(scalarOffsetType))
    width = intType.getWidth();

  SmallVector<HighLevelOffsetLeaf> uniformLeaves;
  SmallVector<HighLevelOffsetLeaf> perLaneLeaves;
  HighLevelUniformityChecker uniformity;
  collectHighLevelOffsetLeaves(offset, APInt(width, 1), ArrayRef<Value>{},
                               ArrayRef<TensorProjection>{}, uniformLeaves,
                               perLaneLeaves, uniformity);

  size_t uBefore = uniformLeaves.size();
  llvm::erase_if(uniformLeaves, isZeroLeaf);
  if (mlirWalkerDebug()) {
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
    llvm::errs() << "[mlir-soffset] fn=" << fnName << "  offset_root="
                 << describeMlirValue(offset) << "  u=" << uniformLeaves.size()
                 << " (was " << uBefore << ")  pl=" << perLaneLeaves.size();
    for (const HighLevelOffsetLeaf &pl : perLaneLeaves)
      llvm::errs() << "  pl_kind=" << describeMlirValue(pl.value);
    llvm::errs() << "\n";
  }
  if (uniformLeaves.empty())
    return {offset, Value()};

  // AMD raw buffer ops bound-check `voffset` without including `soffset`.
  // Lifting a positive uniform component into `soffset` while leaving a
  // possibly-negative one in `voffset` would let some lane's `voffset` go
  // negative, wrap to a huge unsigned, and OOB-drop the access. Only split
  // when every per-lane leaf is provably non-negative.
  auto nonNegative = [&](const HighLevelOffsetLeaf &leaf) {
    return isLeafProvenNonNegative(leaf, solver);
  };
  if (!llvm::all_of(perLaneLeaves, nonNegative)) {
    bool relaxed = false;
    if (relaxedMode == RelaxedNonNegativeMode::NonByte ||
        relaxedMode == RelaxedNonNegativeMode::NonByteLoadToLocal) {
      bool allowLoopCarried =
          relaxedMode == RelaxedNonNegativeMode::NonByteLoadToLocal;
      auto relaxable = [&](const HighLevelOffsetLeaf &leaf) {
        return isRelaxableNonNegativeLeaf(leaf, solver, allowLoopCarried);
      };
      relaxed = llvm::all_of(perLaneLeaves, relaxable);
    } else if (relaxedMode == RelaxedNonNegativeMode::PackedByte &&
               perLaneLeaves.size() > 1) {
      auto relaxable = [&](const HighLevelOffsetLeaf &leaf) {
        return isRelaxablePackedByteLeaf(leaf, solver);
      };
      relaxed = llvm::all_of(perLaneLeaves, relaxable);
    }

    if (!relaxed) {
      if (mlirWalkerDebug())
        llvm::errs() << "[mlir-soffset]   bail: per-lane leaf not provably non-negative\n";
      return {offset, Value()};
    }
    if (mlirWalkerDebug())
      llvm::errs() << "[mlir-soffset]   relaxed: per-lane non-negativity guard\n";
  }

  if (mlirWalkerDebug())
    llvm::errs() << "[mlir-soffset]   -> SPLIT\n";
  Location loc = offset.getLoc();
  SmallVector<Value> uniformValues;
  uniformValues.reserve(uniformLeaves.size());
  for (const HighLevelOffsetLeaf &leaf : uniformLeaves) {
    Value uniform = materializeUniformHighLevelLeaf(leaf, rewriter, loc);
    uniformValues.push_back(
        castScalarInteger(uniform, scalarOffsetType, rewriter, loc));
  }

  SmallVector<Value> perLaneValues;
  perLaneValues.reserve(perLaneLeaves.size());
  for (const HighLevelOffsetLeaf &leaf : perLaneLeaves) {
    perLaneValues.push_back(materializeHighLevelLeaf(
        leaf, rewriter, loc, /*applyProjections=*/true));
  }

  Value uniform = sumIntegerValues(uniformValues, rewriter, loc);
  Value perLane = perLaneValues.empty()
                      ? createZeroTensorLike(offset, rewriter, loc)
                      : sumIntegerValues(perLaneValues, rewriter, loc);
  return {perLane, uniform};
}

bool isByteOffsetSmallerThan2GB(triton::AddPtrOp addPtrOp,
                                std::shared_ptr<DataFlowSolver> solver) {
  Value elemIdx = addPtrOp.getOffset();
  LDBG("Determining value-range of element-index: " << elemIdx);

  // step 1: Get the value range of the element index
  const auto *lattice =
      solver->lookupState<dataflow::IntegerValueRangeLattice>(elemIdx);
  if (!lattice) {
    // Note that it is not always able to get lattice, e.g. the element-index
    // is defined by a tt.load.
    LDBG("Cannot get lattice");
    return false;
  }

  const mlir::IntegerValueRange &vr = lattice->getValue();
  if (vr.isUninitialized() || AMD::isEmptyInitializedRange(vr.getValue())) {
    LDBG("Cannot get value range of the offset");
    return false;
  };

  const auto &smin = vr.getValue().smin();
  const auto &smax = vr.getValue().smax();

  LDBG("Element-index value-range: " << smin << " : " << smax);
  if (smin.isNegative() || smax.isNegative())
    return false;

  // step 2: Get element type and size.
  // e.g. addPtrOp.getType is tensor<64x64x!tt.ptr<f16>, then elemTy is
  // !tt.ptr<f16>, and dereferencing elemTy gets f16.
  // TODO: Not sure if we need to keep dereferencing in a loop.
  Type elemTy = getElementTypeOrSelf(addPtrOp.getType());
  while (auto ptrTy = dyn_cast<triton::PointerType>(elemTy))
    elemTy = ptrTy.getPointeeType();

  if (!elemTy || !elemTy.isIntOrFloat()) {
    LDBG("unknown element type: " << elemTy);
    return false;
  }

  // step 3: check of byte-offset is within 2G
  int64_t elemBitSz = elemTy.getIntOrFloatBitWidth();
  int64_t elemMaxIdx = smax.getSExtValue();
  int64_t byteOfst = (elemBitSz * elemMaxIdx + elemBitSz + 7) / 8;
  int64_t szLimit2GB = (1L << 31) - 1;

  LDBG("element bit sz:" << elemBitSz << ", max byte offset:" << byteOfst
                         << ((szLimit2GB > byteOfst) ? ", out of range"
                                                     : ", in range"));

  return byteOfst <= szLimit2GB;
}

bool isFuncArgWith32bitPtrRange(mlir::Value value) {
  if (value.getDefiningOp())
    return false;

  mlir::BlockArgument blockArg = mlir::cast<mlir::BlockArgument>(value);
  auto blk = blockArg.getOwner();
  auto funcOp = dyn_cast_or_null<tt::FuncOp>(blk->getParentOp());

  if (funcOp && blk == &funcOp->getRegion(0).front()) {
    for (auto [idx, arg] : llvm::enumerate(funcOp.getArguments())) {
      if (arg != value)
        continue;
      auto attr = funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.pointer_range");
      return attr && attr.getInt() <= 32;
    }
  }

  return false;
}

// Pure query: check whether the pointer can be lowered to buffer ops.
// This function must not modify IR. The actual offset truncation (i64 -> i32)
// is handled separately by truncateOffsetToI32().
bool canUseBufferOps(Value ptr,
                     const DenseMap<Value, SetVector<Operation *>> &assumptions,
                     std::shared_ptr<DataFlowSolver> solver,
                     bool analyzeSmallTensorOfst) {
  // 1. Check if the pointer is uniform: i.e., if it comes from a uniform
  // pointer(splatted) and non-uniform offset addition.
  LDBG("Buffer op checks for: " << ptr);
  auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
  if (!addPtrOp)
    return false;

  auto maybeSplatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
  if (!maybeSplatOp)
    return false;
  LDBG("Pattern matched");

  // 2. Check offset bit width. Buffer ops support i32 offsets natively;
  // i64 offsets are truncated later if proven safe.
  Value offset = addPtrOp.getOffset();
  auto ofstBit =
      cast<RankedTensorType>(offset.getType()).getElementTypeBitWidth();
  LLVM_DEBUG(llvm::dbgs() << "offset bits:" << ofstBit << "\n");

  if (ofstBit != 32 && ofstBit != 64)
    return false;

  // 3. Determine if buffer op conversion is safe via pointer_range attribute
  // or range analysis.
  bool isSafe = false;
  if (!analyzeSmallTensorOfst &&
      isFuncArgWith32bitPtrRange(maybeSplatOp.getSrc())) {
    LDBG("base-ptr has tt.pointer_range=32 attribute");
    isSafe = true;
  } else {
    isSafe = isByteOffsetSmallerThan2GB(addPtrOp, std::move(solver));
  }

  return isSafe;
}

// Buffer ops require i32 offsets. If the offset is already i32, return it
// as-is. If it is i64, insert an arith.trunci right before insertBefore.
// The caller's insertion point is saved and restored automatically.
Value truncateOffsetToI32(Value origOffset, OpBuilder &builder, Location loc,
                          Operation *insertBefore) {
  auto offsetTy = cast<RankedTensorType>(origOffset.getType());
  if (offsetTy.getElementTypeBitWidth() == 32)
    return origOffset;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  auto i32Ty = RankedTensorType::get(offsetTy.getShape(), builder.getI32Type(),
                                     offsetTy.getEncoding());
  return arith::TruncIOp::create(builder, loc, i32Ty, origOffset);
}

Value truncateScalarOffsetToI32(Value origOffset, OpBuilder &builder,
                                Location loc, Operation *insertBefore) {
  if (!origOffset)
    return origOffset;
  auto intTy = dyn_cast<IntegerType>(origOffset.getType());
  if (!intTy || intTy.getWidth() == 32)
    return origOffset;
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(insertBefore);
  return arith::TruncIOp::create(builder, loc, builder.getI32Type(),
                                 origOffset);
}

// Extract stride of the blocked offset of LD/ST ops.
Value getBlockStride(Location loc, Value offset, PatternRewriter &rewriter) {
  // Buffer ops take an i32 offset; `truncateOffsetToI32` may insert
  // `arith.trunci` from i64. That op sits in front of the offset chain that
  // `getBlockStride` pattern-matches, so peel it. Any `trunci` here is from
  // that helper (same pass); checking the result is i32 matches it.
  if (auto truncOp = offset.getDefiningOp<arith::TruncIOp>()) {
    if (getElementTypeOrSelf(truncOp.getResult().getType()).isInteger(32))
      offset = truncOp.getIn();
  }
  // canonicalize pointer pass sets block stride via
  // `offset:add-broadcast-muli-splat`, backtrace that pattern to reach the
  // stride.
  if (auto maybeAdd = offset.getDefiningOp<arith::AddIOp>())
    for (auto addOpr : maybeAdd.getOperands())
      if (auto maybeBC = addOpr.getDefiningOp<tt::BroadcastOp>()) {
        auto bcSrc = maybeBC.getSrc();
        if (auto maybeMul = bcSrc.getDefiningOp<arith::MulIOp>())
          for (auto mulOpr : maybeMul.getOperands())
            if (auto maybeSplat = mulOpr.getDefiningOp<tt::SplatOp>())
              return maybeSplat.getSrc();
      }
  return nullptr;
}

// Buffer ops take Optional<I32> stride. getBlockStride walks the offset chain
// and returns the splat's scalar source, which may be i64 when the kernel uses
// i64 indices (e.g. row * stride + col with stride in i64).
static Value maybeTruncateStrideToI32(Value stride, PatternRewriter &rewriter,
                                      Location loc, Operation *insertBefore) {
  if (!stride)
    return stride;
  auto intTy = dyn_cast<IntegerType>(stride.getType());
  if (!intTy)
    return stride;
  if (intTy.getWidth() == 32)
    return stride;
  if (intTy.getWidth() == 64) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(insertBefore);
    return arith::TruncIOp::create(rewriter, loc, rewriter.getI32Type(),
                                   stride);
  }
  return stride;
}

// /*-----------------AtomicCAS-------------------*/

struct ConvertTritonAtomicCASOpToBufferAtomicCAS
    : public mlir::OpRewritePattern<triton::AtomicCASOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicCASOpToBufferAtomicCAS(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::AtomicCASOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicCASOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto sem = op.getSem();
    auto scope = op.getScope();

    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "CAS with unsupported scope");
    }
    LDBG("CAS supported scope");

    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "CAS with unsupported memory ordering");
    }

    // Buffer atomic CAS only supports i32/i64
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "AtomicCAS with unsupported type");
    }
    LDBG("AtomicCAS supported type");

    // All checks passed; now safe to modify IR.
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value offset = addPtrOp.getOffset();
    Value tensorOffset =
        truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    // Buffer atomics support 32 and 64-bit operations, so inputs must be at
    // least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth =
          getVectorSize(basePtr, tensorOffset, axisAnalysisPass) * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(
          op, "BufferAtomicCAS requires opBitWidth >= 32");
    }
    Value blockStride = maybeTruncateStrideToI32(
        getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
        op->getLoc(), op);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicCASOp>(
        op, op.getVal().getType(), basePtr, tensorOffset, op.getCmp(),
        op.getVal(), blockStride, sem, scope);
    return success();
  }

private:
  // Assumptions collected through the function
  const DenseMap<Value, SetVector<Operation *>> &assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct ConvertTritonAtomicRMWOpToBufferAtomicRMW
    : public mlir::OpRewritePattern<triton::AtomicRMWOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonAtomicRMWOpToBufferAtomicRMW(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver,
      const triton::AMD::TargetInfo &targetInfo, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::AtomicRMWOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)), targetInfo(targetInfo),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::AtomicRMWOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();
    auto atomicRmwOp = op.getAtomicRmwOp();
    auto sem = op.getSem();
    auto scope = op.getScope();

    // In addition to the `canUseBufferOps` check, we should ensure that
    // 1. Perform the canUseBufferOps check
    if (!canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      return rewriter.notifyMatchFailure(op, "canUseBufferOps check failed");
    }

    // 2. Check the scope. We support GPU and CTA for now (SYSTEM scope is not
    // supported yet)
    switch (scope) {
    case MemSyncScope::GPU:
    case MemSyncScope::CTA:
      break;
    default:
      return rewriter.notifyMatchFailure(op, "RMW with unsupported scope");
    }
    LDBG("RMW supported scope");

    // 3. Check the memory ordering.
    //    TODO: support monotonic
    switch (sem) {
    case MemSemantic::RELAXED:
    case MemSemantic::RELEASE:
    case MemSemantic::ACQUIRE:
    case MemSemantic::ACQUIRE_RELEASE:
      break;
    default:
      return rewriter.notifyMatchFailure(
          op, "RMW with unsupported memory ordering");
    }

    // 4. Buffer atomic RMW does not support FP8 ops
    //    easier to just check what we support
    auto checkType = getElementTypeOrSelf(op.getVal());
    bool isSupportedType = checkType.isF16() || checkType.isBF16() ||
                           checkType.isF32() || checkType.isF64() ||
                           checkType.isInteger(32) || checkType.isInteger(64);
    if (!isSupportedType) {
      return rewriter.notifyMatchFailure(op, "RMW with unsupported type");
    }
    LDBG("RMW supported type");

    if (atomicRmwOp == RMWOp::FADD &&
        !targetInfo.supportsBufferAtomicFadd(checkType)) {
      return rewriter.notifyMatchFailure(
          op, "RMW FADD unsupported for this type on target");
    }
    LDBG("RMW FADD supported type");

    auto vecSize = getVectorSize(ptr, axisAnalysisPass);
    if (auto mask = op.getMask()) {
      vecSize = std::min(vecSize, axisAnalysisPass.getMaskAlignment(mask));
    }
    // f16/bf16 dtypes could only be efficiently calculated using instructions
    // that pack 2 elements (e.g. @llvm.amdgcn.raw.buffer.atomic.fadd.v2f16)
    if (vecSize % 2 != 0 && (checkType.isF16() || checkType.isBF16())) {
      return rewriter.notifyMatchFailure(
          op, "RMW float 16 dtypes must be aligned by 2");
    }
    LDBG("RMW passed alignment check");

    // 5. Check if the RMWOp is supported
    switch (atomicRmwOp) {
    case RMWOp::AND:
    case RMWOp::OR:
    case RMWOp::XOR:
    case RMWOp::ADD:
    case RMWOp::FADD:
    case RMWOp::UMAX:
    case RMWOp::UMIN:
    case RMWOp::XCHG:
      break;
    case RMWOp::MAX:
    case RMWOp::MIN:
      // TODO: It likely means smax/smin, for now intrinsic
      // llvm.amdgcn.raw.ptr.buffer.atomic.{min|max} is emitted, and llvm get
      // confused as how to deal with {f|s|u}{min|max}.
      if (!checkType.isInteger())
        break;
      // else fall through
    default:
      auto rmwOpStr = stringifyRMWOp(atomicRmwOp).str();
      return rewriter.notifyMatchFailure(op, "RMW with unsupported op: " +
                                                 rmwOpStr);
    }
    LDBG("RMW supported Op");

    // 6. Buffer atomics support 32 and 64-bit operations, so inputs must be at
    //    least 32-bits. Otherwise, fall back to the existing path for atomics
    auto opValueType = op.getVal().getType();
    auto opBitWidth = 0;
    if (auto tensorType = dyn_cast<RankedTensorType>(opValueType)) {
      // We can't just compute the opBitWidth using the numElements *
      // elemBitWidth here. In cases such as tensor<2xf16...>, if the elements
      // are contiguous we can emit the buffer op. Otherwise, the buffer ops
      // lowering will try to emit individual (unsupported) f16/bf16 ops.
      auto elemBitWidth = tensorType.getElementTypeBitWidth();
      opBitWidth = vecSize * elemBitWidth;
    } else {
      opBitWidth = opValueType.getIntOrFloatBitWidth();
    }

    if (opBitWidth < 32) {
      return rewriter.notifyMatchFailure(op, "RMW requires opBitWidth >= 32");
    }

    // All checks passed; now safe to modify IR.
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    Value tensorPtr = addPtrOp.getPtr();
    Value offset = addPtrOp.getOffset();
    Value tensorOffset =
        truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
    auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
    Value basePtr = splatOp.getSrc();

    Value maybeMask{};
    if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
      maybeMask = op.getMask();
    Value blockStride = maybeTruncateStrideToI32(
        getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
        op->getLoc(), op);
    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferAtomicRMWOp>(
        op, op.getVal().getType(), atomicRmwOp, basePtr, tensorOffset,
        op.getVal(), blockStride, sem, scope, maybeMask);

    return success();
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  triton::AMD::TargetInfo targetInfo;
  bool analyzeSmallTensorOfst;
};

// Workaround to allow static_assert(false) on older compilers as it was
// ill-formed before defect report CWG2518
// (https://cplusplus.github.io/CWG/issues/2518.html)
template <typename T> struct always_false : std::false_type {};

template <typename SourceOp>
struct ConvertTritonLoadToBufferLoad : public mlir::OpRewritePattern<SourceOp> {
  using OpRewritePattern<SourceOp>::OpRewritePattern;

  ConvertTritonLoadToBufferLoad(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<SourceOp>(context), assumptions(assumptions),
        axisAnalysisPass(axisAnalysisPass), solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}
  mlir::LogicalResult
  matchAndRewrite(SourceOp op, PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getOperand(0);

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value offset = addPtrOp.getOffset();
      Value tensorOffset =
          truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeOther{};
      if (op.getOther() && !isZeroConst(op.getOther()))
        maybeOther = op.getOther();
      Value maybeMask{};
      if (op.getMask() && !isSplatOneConstTensor(op.getMask()))
        maybeMask = op.getMask();
      Value blockStride = maybeTruncateStrideToI32(
          getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
          op->getLoc(), op);

      auto bufferLoadOp = [&]() {
        if constexpr (std::is_same_v<SourceOp, triton::LoadOp>) {
          unsigned contig = getVectorSize(ptr, axisAnalysisPass);
          if (maybeMask)
            contig = std::min<unsigned>(
                contig, axisAnalysisPass.getMaskAlignment(maybeMask));
          return triton::amdgpu::BufferLoadOp::create(
              rewriter, op->getLoc(), op.getType(), basePtr, tensorOffset,
              /*soffset=*/Value(), blockStride, op.getCache(), maybeMask,
              maybeOther, contig);
        } else if constexpr (std::is_same_v<
                                 SourceOp,
                                 triton::gpu::AsyncCopyGlobalToLocalOp>) {
          return triton::amdgpu::BufferLoadToLocalOp::create(
              rewriter, op->getLoc(), op.getType(), op.getResult(), basePtr,
              tensorOffset, /*soffset=*/Value(), maybeMask, maybeOther,
              blockStride, op.getCache(), op.getContiguity());
        } else {
          static_assert(always_false<SourceOp>::value,
                        "Unsupported type in ConvertTritonLoadToBufferLoad");
        }
      }();

      assert(bufferLoadOp);

      rewriter.replaceOp(op, bufferLoadOp);
      return success();
    }

    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert LoadOp");
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct ConvertTritonStoreToBufferStore
    : public mlir::OpRewritePattern<triton::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  ConvertTritonStoreToBufferStore(
      mlir::MLIRContext *context,
      DenseMap<Value, SetVector<Operation *>> &assumptions,
      ModuleAxisInfoAnalysis &axisAnalysisPass,
      std::shared_ptr<DataFlowSolver> solver, bool analyzeSmallTensorOfst_)
      : mlir::OpRewritePattern<triton::StoreOp>(context),
        assumptions(assumptions), axisAnalysisPass(axisAnalysisPass),
        solver(std::move(solver)),
        analyzeSmallTensorOfst(analyzeSmallTensorOfst_) {}

  mlir::LogicalResult
  matchAndRewrite(triton::StoreOp op,
                  PatternRewriter &rewriter) const override {
    LDBG("Try to convert: " << op);
    Value ptr = op.getPtr();

    if (canUseBufferOps(ptr, assumptions, solver, analyzeSmallTensorOfst)) {
      auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
      Value tensorPtr = addPtrOp.getPtr();
      Value offset = addPtrOp.getOffset();
      Value tensorOffset =
          truncateOffsetToI32(offset, rewriter, op->getLoc(), op);
      auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>();
      Value basePtr = splatOp.getSrc();
      Value maybeMask{};
      unsigned contig = getVectorSize(ptr, axisAnalysisPass);
      if (op.getMask() && !isSplatOneConstTensor(op.getMask())) {
        maybeMask = op.getMask();
        contig = std::min<unsigned>(
            contig, axisAnalysisPass.getMaskAlignment(maybeMask));
      }
      Value blockStride = maybeTruncateStrideToI32(
          getBlockStride(op->getLoc(), tensorOffset, rewriter), rewriter,
          op->getLoc(), op);

      rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
          op, op.getValue(), basePtr, tensorOffset, /*soffset=*/Value(),
          blockStride, op.getCache(), maybeMask, contig);
      return success();
    }
    LDBG("Failed to convert: " << op);
    return rewriter.notifyMatchFailure(op, "Failed to convert StoreOp");
  }

private:
  // Assumptions collected through the function
  DenseMap<Value, SetVector<Operation *>> assumptions;
  ModuleAxisInfoAnalysis &axisAnalysisPass;
  std::shared_ptr<DataFlowSolver> solver;
  bool analyzeSmallTensorOfst;
};

struct SplitBufferLoadOffset
    : public mlir::OpRewritePattern<triton::amdgpu::BufferLoadOp> {
  SplitBufferLoadOffset(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(triton::amdgpu::BufferLoadOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSoffset())
      return failure();

    auto [voffset, soffset] =
        splitHighLevelBufferOffset(
            op.getVoffset(), rewriter, solver,
            relaxedNonNegativeSplit() && hasNonByteElement(op.getType())
                ? RelaxedNonNegativeMode::NonByte
                : RelaxedNonNegativeMode::None);
    if (!soffset)
      return failure();

    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferLoadOp>(
        op, op.getType(), op.getPtr(), voffset, soffset, op.getStride(),
        op.getCache(), op.getMask(), op.getOther(), op.getContiguity());
    return success();
  }

private:
  DataFlowSolver *solver;
};

struct SplitBufferLoadToLocalOffset
    : public mlir::OpRewritePattern<triton::amdgpu::BufferLoadToLocalOp> {
  SplitBufferLoadToLocalOffset(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(triton::amdgpu::BufferLoadToLocalOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSoffset())
      return failure();

    auto [voffset, soffset] =
        splitHighLevelBufferOffset(
            op.getVoffset(), rewriter, solver,
            relaxedNonNegativeSplit()
                ? (hasNonByteElement(op.getDest().getType())
                       ? RelaxedNonNegativeMode::NonByteLoadToLocal
                       : (hasByteElement(op.getDest().getType())
                              ? RelaxedNonNegativeMode::PackedByte
                              : RelaxedNonNegativeMode::None))
                : RelaxedNonNegativeMode::None);
    if (!soffset)
      return failure();

    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferLoadToLocalOp>(
        op, op.getType(), op.getDest(), op.getPtr(), voffset, soffset,
        op.getMask(), op.getOther(), op.getStride(), op.getCache(),
        op.getContiguity());
    return success();
  }

private:
  DataFlowSolver *solver;
};

struct SplitBufferStoreOffset
    : public mlir::OpRewritePattern<triton::amdgpu::BufferStoreOp> {
  SplitBufferStoreOffset(MLIRContext *context, DataFlowSolver *solver)
      : OpRewritePattern(context), solver(solver) {}

  LogicalResult matchAndRewrite(triton::amdgpu::BufferStoreOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSoffset())
      return failure();

    auto [voffset, soffset] =
        splitHighLevelBufferOffset(
            op.getVoffset(), rewriter, solver,
            relaxedNonNegativeSplit() && hasNonByteElement(op.getValue().getType())
                ? RelaxedNonNegativeMode::NonByte
                : RelaxedNonNegativeMode::None);
    if (!soffset)
      return failure();

    rewriter.replaceOpWithNewOp<triton::amdgpu::BufferStoreOp>(
        op, op.getValue(), op.getPtr(), voffset, soffset, op.getStride(),
        op.getCache(), op.getMask(), op.getContiguity());
    return success();
  }

private:
  DataFlowSolver *solver;
};

} // anonymous namespace

struct TritonAMDGPUConvertToBufferOpsPass
    : impl::TritonAMDGPUConvertToBufferOpsBase<
          TritonAMDGPUConvertToBufferOpsPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ModuleOp mod = getOperation();
    triton::AMD::TargetInfo targetInfo(gfxArch);

    // Collect assumptions in the function
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(getOperation());
    std::shared_ptr<DataFlowSolver> solver = createDataFlowSolver();

    AMD::TritonIntegerRangeAnalysis *rangeAnalysis =
        solver->load<AMD::TritonIntegerRangeAnalysis>(
            assumptions, &getAnalysis<DominanceInfo>());
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(getOperation())))
      return signalPassFailure();

    AMD::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    patterns.add<ConvertTritonLoadToBufferLoad<tt::LoadOp>,
                 ConvertTritonStoreToBufferStore>(context, assumptions,
                                                  axisInfoAnalysis, solver,
                                                  this->analyzeSmallTensorOfst);
    if (targetInfo.supportsBufferLoadToLocal()) {
      patterns
          .add<ConvertTritonLoadToBufferLoad<ttg::AsyncCopyGlobalToLocalOp>>(
              context, assumptions, axisInfoAnalysis, solver,
              this->analyzeSmallTensorOfst);
    }

    if (this->allowBufferAtomics && targetInfo.supportsBufferAtomicRMW())
      patterns.add<ConvertTritonAtomicRMWOpToBufferAtomicRMW>(
          context, assumptions, axisInfoAnalysis, solver, targetInfo,
          this->analyzeSmallTensorOfst);
    patterns.add<ConvertTritonAtomicCASOpToBufferAtomicCAS>(
        context, assumptions, axisInfoAnalysis, solver,
        this->analyzeSmallTensorOfst);

    if (applyPatternsGreedily(mod, std::move(patterns)).failed())
      signalPassFailure();

    // Lift wave-uniform addends of each AMD buffer op's `voffset`
    // into its scalar `soffset` operand so uniform address movement is routed
    // through the SGPR slot of the raw buffer intrinsic instead of consuming
    // VGPRs and per-lane VALU adds. Running this after the conversion patterns
    // above leaves only `amdgpu.buffer_*` shapes for these matchers to handle.
    RewritePatternSet splitPatterns(context);
    splitPatterns.add<SplitBufferLoadOffset, SplitBufferLoadToLocalOffset,
                      SplitBufferStoreOffset>(context, solver.get());
    if (applyPatternsGreedily(mod, std::move(splitPatterns)).failed())
      signalPassFailure();
  }
};

} // namespace mlir
