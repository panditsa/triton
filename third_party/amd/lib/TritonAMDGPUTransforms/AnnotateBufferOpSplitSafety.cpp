#include "Analysis/RangeAnalysis.h"
#include "TritonAMDGPUTransforms/Passes.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {

#define GEN_PASS_DEF_TRITONAMDGPUANNOTATEBUFFEROPSPLITSAFETY
#include "TritonAMDGPUTransforms/Passes.h.inc"

namespace {

namespace AMD = mlir::triton::AMD;
namespace tta = mlir::triton::amdgpu;
namespace ttg = mlir::triton::gpu;

constexpr llvm::StringLiteral kSplitSafeAttrName = "amdgpu.split_soffset_safe";
constexpr llvm::StringLiteral kSplitUnsafeAttrName =
    "amdgpu.split_soffset_unsafe_reason";

// Shape/layout ops that forward their operand's values unchanged, and with them
// the operand's sign and its integer-range lattice state.
static bool isTransparentWrapper(Operation *op) {
  bool isWrapper =
      isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
          triton::ReshapeOp, ttg::ConvertLayoutOp>(op);
  assert((!isWrapper || op->getNumOperands() == 1) &&
         "transparent wrapper must have a single SSA operand.");
  return isWrapper;
}

// Peel transparent wrappers to expose the real defining op underneath.
static Value peelTransparentWrappers(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (!isTransparentWrapper(def))
      break;
    v = def->getOperand(0);
  }
  return v;
}

// Conservatively accept an offset only when every leaf in its
// additive/shape expression proves non-negative. This may miss safe splits,
// but never annotates an offset with a possibly-negative voffset.
static bool isLeafNonNegative(Value v, DataFlowSolver &solver) {
  // An `add` is never a leaf to the soffset splitter. It peels the summands
  // apart and lifts the uniform ones into the unsigned soffset. So a sum whose
  // range is non-negative can still hide a negative summand.
  if (peelTransparentWrappers(v).getDefiningOp<arith::AddIOp>())
    return false;

  const auto *range = solver.lookupState<dataflow::IntegerValueRangeLattice>(v);
  if (!range || range->getValue().isUninitialized())
    return false;
  if (AMD::isEmptyInitializedRange(range->getValue().getValue()))
    return false;
  return succeeded(dataflow::staticallyNonNegative(solver, v));
}

static std::string describeValueForSplitSafety(Value v, DataFlowSolver &solver) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  if (!v) {
    os << "<null>";
    return os.str();
  }

  Value peeled = peelTransparentWrappers(v);
  os << "value=";
  peeled.print(os);
  os << " type=" << peeled.getType();

  if (Operation *def = peeled.getDefiningOp())
    os << " def=" << def->getName().getStringRef();
  else if (auto arg = dyn_cast<BlockArgument>(peeled))
    os << " block_arg=" << arg.getArgNumber();
  else
    os << " def=<none>";

  const auto *range = solver.lookupState<dataflow::IntegerValueRangeLattice>(v);
  if (!range) {
    os << " range=<missing>";
  } else if (range->getValue().isUninitialized()) {
    os << " range=<uninitialized>";
  } else if (AMD::isEmptyInitializedRange(range->getValue().getValue())) {
    os << " range=<empty-initialized>";
  } else {
    const auto &bounds = range->getValue().getValue();
    os << " range.s=[" << bounds.smin() << "," << bounds.smax() << "]";
    os << " range.u=[" << bounds.umin() << "," << bounds.umax() << "]";
  }
  return os.str();
}

// `arith.trunci` preserves non-negativity only when the source value fits in
// the signed non-negative range of the destination. Otherwise a positive i64
// such as 2^31 truncates to a negative i32.
static bool isTruncINonNegative(arith::TruncIOp truncOp,
                                 DataFlowSolver &solver,
                                 std::string *reason) {
  const auto *range = solver.lookupState<dataflow::IntegerValueRangeLattice>(
      truncOp.getIn());
  if (!range || range->getValue().isUninitialized()) {
    if (reason)
      *reason = "arith.trunci source range unavailable: " +
                describeValueForSplitSafety(truncOp.getIn(), solver);
    return false;
  }
  if (AMD::isEmptyInitializedRange(range->getValue().getValue())) {
    if (reason)
      *reason = "arith.trunci source range empty-initialized: " +
                describeValueForSplitSafety(truncOp.getIn(), solver);
    return false;
  }

  auto dstTy = dyn_cast<IntegerType>(getElementTypeOrSelf(truncOp.getOut()));
  if (!dstTy) {
    if (reason)
      *reason = "arith.trunci result is not integer: " +
                describeValueForSplitSafety(truncOp.getOut(), solver);
    return false;
  }

  const auto &bounds = range->getValue().getValue();
  if (bounds.smin().isNegative()) {
    if (reason)
      *reason = "arith.trunci source may be negative: " +
                describeValueForSplitSafety(truncOp.getIn(), solver);
    return false;
  }

  unsigned srcWidth = bounds.smax().getBitWidth();
  llvm::APInt dstSignedMax =
      llvm::APInt::getSignedMaxValue(dstTy.getWidth()).zextOrTrunc(srcWidth);
  if (bounds.smax().sgt(dstSignedMax)) {
    if (reason)
      *reason = "arith.trunci source may exceed destination signed max: " +
                describeValueForSplitSafety(truncOp.getIn(), solver);
    return false;
  }
  return true;
}

static bool isNonNegativeImpl(Value v, DataFlowSolver &solver,
                              std::string *reason, unsigned depth = 0) {
  if (!v) {
    if (reason)
      *reason = "null value";
    return false;
  }

  // Prefer the lattice result. The structural cases below enforce the
  // stronger leaf-wise proof needed before splitting soffset from voffset.
  if (isLeafNonNegative(v, solver))
    return true;

  Operation *def = v.getDefiningOp();
  if (!def) {
    if (reason)
      *reason = "no defining op: " + describeValueForSplitSafety(v, solver);
    return false;
  }

  // Recurse through ops where "all operands non-negative -> result
  // non-negative" (with the same < 2GB wrap caveat the rest of the
  // buffer-op path already accepts on add/mul).
  if (isa<arith::AddIOp, arith::MulIOp, arith::OrIOp, arith::XOrIOp,
          arith::DivSIOp, arith::DivUIOp, arith::MinSIOp, arith::MinUIOp,
          arith::MaxSIOp, arith::MaxUIOp, arith::ExtSIOp>(def)) {
    unsigned idx = 0;
    for (Value operand : def->getOperands()) {
      std::string childReason;
      if (!isNonNegativeImpl(operand, solver, &childReason, depth + 1)) {
        if (reason) {
          std::string storage;
          llvm::raw_string_ostream os(storage);
          os << "op " << def->getName().getStringRef() << " operand " << idx
             << " not non-negative; " << childReason;
          *reason = os.str();
        }
        return false;
      }
      ++idx;
    }
    return true;
  }

  if (auto trunc = dyn_cast<arith::TruncIOp>(def))
    return isTruncINonNegative(trunc, solver, reason);

  // First operand only (sign carries from operand 0).
  if (isa<arith::ShLIOp, arith::ShRSIOp, arith::RemSIOp, arith::RemUIOp>(def))
    return isNonNegativeImpl(def->getOperand(0), solver, reason, depth + 1);

  // Always non-negative regardless of operands.
  if (isa<arith::ShRUIOp, arith::ExtUIOp>(def))
    return true;

  // Triton shape/control ops that are non-negative or preserve sign.
  if (auto mr = dyn_cast<triton::MakeRangeOp>(def)) {
    bool ok = mr.getStartAttr().getInt() >= 0;
    if (!ok && reason)
      *reason = "tt.make_range starts negative: " +
                describeValueForSplitSafety(v, solver);
    return ok;
  }
  if (isa<triton::GetProgramIdOp, triton::GetNumProgramsOp>(def))
    return true;
  if (isTransparentWrapper(def))
    return isNonNegativeImpl(def->getOperand(0), solver, reason, depth + 1);

  if (reason) {
    std::string storage;
    llvm::raw_string_ostream os(storage);
    os << "unsupported/non-proven op " << def->getName().getStringRef() << "; "
       << describeValueForSplitSafety(v, solver);
    *reason = os.str();
  }
  return false;
}

static bool isNonNegative(Value v, DataFlowSolver &solver,
                          std::string *reason = nullptr) {
  return isNonNegativeImpl(v, solver, reason);
}

struct AnnotateBufferOpSplitSafetyPass
    : impl::TritonAMDGPUAnnotateBufferOpSplitSafetyBase<
          AnnotateBufferOpSplitSafetyPass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Reuse AMD integer range analysis so `tl.assume` / `gl.assume` and
    // argument attributes can prove buffer offsets non-negative.
    DenseMap<Value, SetVector<Operation *>> assumptions =
        AMD::TritonIntegerRangeAnalysis::collectAssumptions(mod);
    auto solver = createDataFlowSolver();
    auto *rangeAnalysis = solver->load<AMD::TritonIntegerRangeAnalysis>(
        assumptions, &getAnalysis<DominanceInfo>());
    AMD::initializeFuncOps(mod, rangeAnalysis);
    if (failed(solver->initializeAndRun(mod)))
      return signalPassFailure();

    UnitAttr unit = UnitAttr::get(&getContext());
    auto annotateIfSafe = [&](Operation *op, Value offsets) {
      std::string unsafeReason;
      if (isNonNegative(offsets, *solver, &unsafeReason)) {
        op->setAttr(kSplitSafeAttrName, unit);
        op->removeAttr(kSplitUnsafeAttrName);
      } else {
        op->setAttr(kSplitUnsafeAttrName,
                    StringAttr::get(&getContext(), unsafeReason));
      }
    };

    mod.walk([&](Operation *op) {
      if (auto load = dyn_cast<tta::BufferLoadOp>(op))
        annotateIfSafe(op, load.getOffsets());
      else if (auto store = dyn_cast<tta::BufferStoreOp>(op))
        annotateIfSafe(op, store.getOffsets());
      else if (auto loadLds = dyn_cast<tta::BufferLoadToLocalOp>(op))
        annotateIfSafe(op, loadLds.getOffsets());
    });
  }
};

} // namespace
} // namespace mlir
