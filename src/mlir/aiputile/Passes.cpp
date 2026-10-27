#include "aiputile/Passes.h"
#include "aiputile/AIPUTileOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <limits>

using namespace mlir;

namespace {
// Match by the public operation names so this library does not depend on the
// proprietary graph builder or its quantization implementation. In aipugc these
// are the registered, verified source operations, not a second source dialect.
struct ImportBinary : ConversionPattern {
  ImportBinary(StringRef source, MLIRContext *ctx)
      : ConversionPattern(source, 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumRegions() ||
        failed(aiputile::verifyTensorTypes(op)))
      return failure();
    for (NamedAttribute attr : op->getAttrs()) {
      // The source builder flattens optional parameters onto the operation.
      if (attr.getName() == "dictAttr") {
        if (auto dict = dyn_cast<DictionaryAttr>(attr.getValue());
            dict && dict.empty())
          continue;
      }
      if (attr.getName() == "with_activation") {
        if (auto value = dyn_cast<StringAttr>(attr.getValue());
            value && value.getValue() == "NONE")
          continue;
      }
      return op->emitError("tile path does not support source parameter ")
             << attr.getName();
    }
    if (op->getName().getStringRef() == "aipugc.add")
      rewriter.replaceOpWithNewOp<aiputile::AddOp>(op, op->getResultTypes(),
                                                   operands[0], operands[1]);
    else
      rewriter.replaceOpWithNewOp<aiputile::MulOp>(op, op->getResultTypes(),
                                                   operands[0], operands[1]);
    return success();
  }
};

struct ImportFunction : ConversionPattern {
  ImportFunction(MLIRContext *ctx) : ConversionPattern("aipugc.func", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = op->getAttrOfType<TypeAttr>("function_type");
    auto name = op->getAttrOfType<StringAttr>("sym_name");
    if (!type || !isa<FunctionType>(type.getValue()) || !name ||
        op->getNumRegions() != 1)
      return op->emitError("expected a well-formed aipugc.func");
    auto func = rewriter.create<func::FuncOp>(
        op->getLoc(), name.getValue(), cast<FunctionType>(type.getValue()));
    func->setAttrs(op->getAttrs());
    rewriter.inlineRegionBefore(op->getRegion(0), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ImportReturn : ConversionPattern {
  ImportReturn(MLIRContext *ctx) : ConversionPattern("aipugc.return", 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, operands);
    return success();
  }
};

struct ImportPass : PassWrapper<ImportPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ImportPass)
  StringRef getArgument() const final { return "lower-aipugc-to-aiputile"; }
  StringRef getDescription() const final {
    return "Import static dense FP32 AIPUGC Add/Mul without TOSA or Linalg";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aiputile::AIPUTileDialect, func::FuncDialect>();
  }
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getName().getDialectNamespace() != "aipugc";
    });
    RewritePatternSet patterns(&getContext());
    patterns.add<ImportBinary>("aipugc.add", &getContext());
    patterns.add<ImportBinary>("aipugc.mul", &getContext());
    patterns.add<ImportFunction, ImportReturn>(&getContext());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      getOperation().emitError(
          "tile path supports only static dense FP32 Add/Mul and "
          "function/return; unsupported source IR remains");
      signalPassFailure();
    }
  }
};

struct FuseAddMul : OpRewritePattern<aiputile::MulOp> {
  FuseAddMul(MLIRContext *ctx, int64_t size)
      : OpRewritePattern(ctx), tileSize(size) {}
  LogicalResult matchAndRewrite(aiputile::MulOp mul,
                                PatternRewriter &rewriter) const override {
    auto add = mul.getLhs().getDefiningOp<aiputile::AddOp>();
    bool addOnLhs = true;
    if (!add) {
      add = mul.getRhs().getDefiningOp<aiputile::AddOp>();
      addOnLhs = false;
    }
    if (!add || !add.getOutput().hasOneUse() ||
        add->getBlock() != mul->getBlock())
      return rewriter.notifyMatchFailure(
          mul, "requires a same-block single-consumer Add producer");
    rewriter.replaceOpWithNewOp<aiputile::FusedAddMulOp>(
        mul, mul.getOutput().getType(), add.getLhs(), add.getRhs(),
        addOnLhs ? mul.getRhs() : mul.getLhs(),
        rewriter.getI64IntegerAttr(tileSize), rewriter.getBoolAttr(addOnLhs));
    rewriter.eraseOp(add);
    return success();
  }
  int64_t tileSize;
};

struct FusionPass : PassWrapper<FusionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FusionPass)
  FusionPass() = default;
  FusionPass(const FusionPass &other) : PassWrapper(other) {}
  explicit FusionPass(int64_t size) { tileSize = size; }
  Option<int64_t> tileSize{*this, "tile-size",
                           llvm::cl::desc("Elements per logical tile"),
                           llvm::cl::init(256)};
  StringRef getArgument() const final { return "aiputile-fuse-add-mul"; }
  StringRef getDescription() const final {
    return "Fuse a single-use Add into Mul, retaining operand order";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<aiputile::AIPUTileDialect>();
  }
  void runOnOperation() override {
    if (tileSize <= 0 || tileSize > std::numeric_limits<int32_t>::max()) {
      getOperation().emitError("tile-size must be in [1, INT32_MAX]");
      return signalPassFailure();
    }
    // Check before the greedy driver's implicit DCE: even a currently dead
    // second consumer must not make the initial fusion eligibility ambiguous.
    bool sharedProducer = false;
    getOperation().walk([&](aiputile::AddOp add) {
      if (!add.getOutput().hasOneUse()) {
        add.emitError("cannot fuse: requires a single-consumer Add result");
        sharedProducer = true;
      }
    });
    if (sharedProducer)
      return signalPassFailure();
    RewritePatternSet patterns(&getContext());
    patterns.add<FuseAddMul>(&getContext(), tileSize);
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
#if defined(AIPUTILE_HAS_APPLY_PATTERNS_GREEDILY)
    if (failed(applyPatternsGreedily(getOperation(), frozenPatterns)))
#else
    if (failed(applyPatternsAndFoldGreedily(getOperation(), frozenPatterns)))
#endif
      return signalPassFailure();
    bool remaining = false;
    getOperation().walk([&](Operation *op) {
      if (isa<aiputile::AddOp, aiputile::MulOp>(op)) {
        op->emitError("cannot fuse: requires Add -> Mul with a same-block, "
                      "single-consumer Add result");
        remaining = true;
      }
    });
    if (remaining)
      signalPassFailure();
  }
};

// Preflight the complete function before mutating signatures. No calls, hidden
// effects, aliases of tensor views, or intermediate results may be discarded.
LogicalResult checkKernel(func::FuncOp func, ModuleOp module) {
  if (func.isDeclaration() || !llvm::hasSingleElement(func.getBody()) ||
      func.getNumArguments() != 3 || func.getNumResults() != 1 ||
      std::distance(func.front().begin(), func.front().end()) != 2)
    return func.emitError("tile kernel requires one block, three runtime "
                          "tensor inputs, one fused operation and one return");
  auto fused = dyn_cast<aiputile::FusedAddMulOp>(func.front().front());
  auto ret = dyn_cast<func::ReturnOp>(func.front().back());
  if (!fused || !ret || ret.getNumOperands() != 1 ||
      ret.getOperand(0) != fused.getOutput())
    return func.emitError(
        "tile kernel must return its fused Add/Mul result directly");
  for (Type type : func.getArgumentTypes())
    if (type != fused.getOutput().getType())
      return func.emitError(
          "all kernel arguments must match the dense static FP32 result");
  for (Value input : fused->getOperands()) {
    auto arg = dyn_cast<BlockArgument>(input);
    if (!arg || arg.getOwner() != &func.front())
      return fused.emitError("requires runtime function arguments, not "
                             "constants, views or intermediate tensors");
  }
  for (NamedAttribute attr : func->getAttrs()) {
    auto name = attr.getName().strref();
    if (name != "sym_name" && name != "function_type" &&
        name != "sym_visibility")
      return func.emitError("unsupported function/argument/result attributes "
                            "on tile kernel: ")
             << name;
  }
  auto uses = SymbolTable::getSymbolUses(func, module);
  if (!uses || !uses->empty())
    return func.emitError("tile output-parameter ABI requires an entry "
                          "function with no in-module callers");
  return success();
}

void lowerKernel(func::FuncOp oldFunc, unsigned addressSpace) {
  auto fused = cast<aiputile::FusedAddMulOp>(oldFunc.front().front());
  auto tensor = cast<RankedTensorType>(fused.getOutput().getType());
  SmallVector<unsigned> inputArgs;
  for (Value value : fused->getOperands())
    inputArgs.push_back(cast<BlockArgument>(value).getArgNumber());
  const bool addOnLhs = fused.getAddOnLhs();
  const int64_t n = tensor.getNumElements();
  const int64_t tileSize = std::min<int64_t>(fused.getTileSize(), n);
  const int64_t tiles = 1 + (n - 1) / tileSize;
  OpBuilder builder(oldFunc);
  auto memory = builder.getI64IntegerAttr(addressSpace);
  auto memref = MemRefType::get(tensor.getShape(), tensor.getElementType(),
                                MemRefLayoutAttrInterface{}, memory);
  SmallVector<Type> args(4, memref);
  auto func =
      builder.create<func::FuncOp>(oldFunc.getLoc(), oldFunc.getSymName(),
                                   builder.getFunctionType(args, {}));
  if (auto visibility = oldFunc->getAttr("sym_visibility"))
    func->setAttr("sym_visibility", visibility);
  // This is an output-parameter entry ABI. The caller provides a disjoint,
  // contiguous output allocation; no temporary full-size tensor is allocated.
  func.setArgAttr(3, "llvm.noalias", builder.getUnitAttr());
  func->setAttr("aiputile.tile_size", builder.getI64IntegerAttr(tileSize));
  // Logical staging payload: X, B, A, temporary and Y, all FP32. This excludes
  // alignment, DMA descriptors and runtime overhead; it is not an SRAM limit.
  // No staging allocation or physical SRAM mapping is performed in this pass.
  func->setAttr("aiputile.staging_bytes",
                builder.getI64IntegerAttr(5 * tileSize * 4));
  Block *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Location loc = func.getLoc();
  auto index = [&](int64_t value) -> Value {
    return builder.create<arith::ConstantIndexOp>(loc, value);
  };
  Value zero = index(0), one = index(1), step = index(tileSize),
        count = index(n);
  Value tileCount = index(tiles);
  builder.create<scf::ForOp>(
      loc, zero, tileCount, one, ValueRange{},
      [&](OpBuilder &outer, Location loc, Value tile, ValueRange) {
        Value offset = outer.create<arith::MulIOp>(loc, tile, step);
        Value remaining = outer.create<arith::SubIOp>(loc, count, offset);
        Value length = outer.create<arith::MinUIOp>(loc, remaining, step);
        outer.create<scf::ForOp>(
            loc, zero, length, one, ValueRange{},
            [&](OpBuilder &inner, Location loc, Value local, ValueRange) {
              Value flat = inner.create<arith::AddIOp>(loc, offset, local);
              SmallVector<Value> indices(tensor.getRank());
              Value quotient = flat;
              for (int64_t d = tensor.getRank() - 1; d >= 0; --d) {
                Value size = inner.create<arith::ConstantIndexOp>(
                    loc, tensor.getDimSize(d));
                indices[d] = inner.create<arith::RemUIOp>(loc, quotient, size);
                if (d != 0)
                  quotient = inner.create<arith::DivUIOp>(loc, quotient, size);
              }
              SmallVector<Value> inputs;
              for (unsigned arg : inputArgs)
                inputs.push_back(inner.create<memref::LoadOp>(
                    loc, entry->getArgument(arg), indices));
              Value tmp =
                  inner.create<arith::AddFOp>(loc, inputs[0], inputs[1]);
              Value result = inner.create<arith::MulFOp>(
                  loc, addOnLhs ? tmp : inputs[2], addOnLhs ? inputs[2] : tmp);
              inner.create<memref::StoreOp>(loc, result, entry->getArgument(3),
                                            indices);
              inner.create<scf::YieldOp>(loc);
            });
        outer.create<scf::YieldOp>(loc);
      });
  builder.create<func::ReturnOp>(loc);
  oldFunc.erase();
}

struct TileLoweringPass
    : PassWrapper<TileLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileLoweringPass)
  TileLoweringPass() = default;
  TileLoweringPass(const TileLoweringPass &other) : PassWrapper(other) {}
  explicit TileLoweringPass(unsigned space) { addressSpace = space; }
  Option<unsigned> addressSpace{
      *this, "device-address-space",
      llvm::cl::desc("ABI memory space (0 host, 1 Zhouyi device; not SRAM)"),
      llvm::cl::init(0)};
  StringRef getArgument() const final { return "aiputile-lower-tiles"; }
  StringRef getDescription() const final {
    return "Expand fused tensor kernels into output-parameter SCF tile loops";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, memref::MemRefDialect,
                    arith::ArithDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    if (addressSpace > 1) {
      getOperation().emitError(
          "supported ABI address spaces are 0 (host) and 1 (Zhouyi device)");
      return signalPassFailure();
    }
    SmallVector<func::FuncOp> funcs;
    for (Operation &op : getOperation().getBody()->getOperations()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func || failed(checkKernel(func, getOperation()))) {
        if (!func)
          op.emitError("tile lowering only accepts entry functions");
        return signalPassFailure();
      }
      funcs.push_back(func);
    }
    if (funcs.empty()) {
      getOperation().emitError("expected at least one tile kernel");
      return signalPassFailure();
    }
    for (auto func : funcs)
      lowerKernel(func, addressSpace);
  }
};

struct TileToLLVMPass : PassWrapper<TileToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TileToLLVMPass)
  TileToLLVMPass() = default;
  TileToLLVMPass(const TileToLLVMPass &other) : PassWrapper(other) {}
  TileToLLVMPass(bool bare, unsigned width) {
    barePointers = bare;
    indexBitwidth = width;
  }
  Option<bool> barePointers{*this, "bare-pointers", llvm::cl::init(true)};
  Option<unsigned> indexBitwidth{*this, "index-bitwidth", llvm::cl::init(64)};
  StringRef getArgument() const final { return "aiputile-to-llvm"; }
  StringRef getDescription() const final {
    return "Lower SCF/Arith/MemRef/Func to LLVM with a consistent pointer ABI";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, cf::ControlFlowDialect>();
  }
  void runOnOperation() override {
    if (indexBitwidth != 32 && indexBitwidth != 64) {
      getOperation().emitError("index-bitwidth must be 32 or 64");
      return signalPassFailure();
    }
    LowerToLLVMOptions options(&getContext());
    options.useBarePtrCallConv = barePointers;
    options.overrideIndexBitwidth(indexBitwidth);
    LLVMTypeConverter converter(&getContext(), options);
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    RewritePatternSet patterns(&getContext());
    populateSCFToControlFlowConversionPatterns(patterns);
    arith::populateArithToLLVMConversionPatterns(converter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    populateFuncToLLVMConversionPatterns(converter, patterns);
    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> aiputile::createLowerAipugcToAIPUTilePass() {
  return std::make_unique<ImportPass>();
}
std::unique_ptr<Pass> aiputile::createTileFusionPass(int64_t size) {
  return std::make_unique<FusionPass>(size);
}
std::unique_ptr<Pass> aiputile::createTileLoweringPass(unsigned space) {
  return std::make_unique<TileLoweringPass>(space);
}
std::unique_ptr<Pass> aiputile::createTileToLLVMPass(bool bare,
                                                     unsigned width) {
  return std::make_unique<TileToLLVMPass>(bare, width);
}
void aiputile::registerAIPUTilePasses() {
  PassRegistration<ImportPass>();
  PassRegistration<FusionPass>();
  PassRegistration<TileLoweringPass>();
  PassRegistration<TileToLLVMPass>();
}
