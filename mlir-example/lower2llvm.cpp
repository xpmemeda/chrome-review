#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Conversion/LLVMCommon/VectorPattern.h"

#include "dialect.h"
#include "passes.h"

using namespace mlir;

namespace {

struct ScalarF64AddOpLowering : public ConversionPattern {
  ScalarF64AddOpLowering(MLIRContext *ctx)
      : ConversionPattern(demo::ScalarF64AddOp::getOperationName(), 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::FAddOp>(op, operands);
    return success();
  }
};

struct ScalarF64MulOpLowering : public ConversionPattern {
  ScalarF64MulOpLowering(MLIRContext *ctx)
      : ConversionPattern(demo::ScalarF64MulOp::getOperationName(), 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    rewriter.replaceOpWithNewOp<LLVM::FMulOp>(op, operands);
    return success();
  }
};

using PointWiseNegOpLowering = VectorConvertToLLVMPattern<demo::PointWiseNegOp, LLVM::FNegOp>;
using PointWiseExpOpLowering = VectorConvertToLLVMPattern<demo::PointWiseExpOp, LLVM::ExpOp>;

struct EmitDoubleNegPattern : public mlir::RewritePattern {
  EmitDoubleNegPattern(mlir::MLIRContext *ctx)
      : mlir::RewritePattern(demo::PointWiseNegOp::getOperationName(), 1, ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, mlir::PatternRewriter &rewriter) const final {
    // Note: get next or prev op
    // llvm::errs() << "XXX: " << mlir::isa<demo::PointWiseNegOp>(*op->getResult(0).user_begin()) << "\n";
    // llvm::errs() << "XXX: " << mlir::isa<demo::PointWiseNegOp>(op->getOperand(0).getDefiningOp()) << "\n";

    // auto next = *op->getResult(0).user_begin();
    // if (mlir::isa<demo::PointWiseNegOp>(next)) {
    //   rewriter.replaceOp(next, {op->getOperand(0)});  // err: expected pattern to replace the root operation
    // }

    // auto prev = op->getOperand(0).getDefiningOp();
    // if (prev && mlir::isa<demo::PointWiseNegOp>(prev)) {
    //   rewriter.replaceOp(op, {prev->getOperand(0)});  // err: expected pattern to replace the root operation
    // }

    return mlir::success();
  }
};

struct MatmulOpLoweringToLLVM : public ConvertOpToLLVMPattern<demo::MatmulOp> {
  using ConvertOpToLLVMPattern<demo::MatmulOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(demo::MatmulOp op, demo::MatmulOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // IntegerAttr m = rewriter.getI32IntegerAttr(128);
    rewriter.replaceOpWithNewOp<mlir::LLVM::MatrixMultiplyOp>(
        op, typeConverter->convertType(op->getResult(0).getType()),
        adaptor.lhs(), adaptor.rhs(), 3, 3, 3);
    return success();
  }
};

struct MatmulOpLoweringToVector : public ConversionPattern {
  MatmulOpLoweringToVector(MLIRContext *ctx)
      : ConversionPattern(demo::MatmulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto adaptor = demo::MatmulOp::Adaptor(operands);
    rewriter.replaceOpWithNewOp<mlir::vector::MatmulOp>(
        op, adaptor.lhs(), adaptor.rhs(), 3, 3, 3);
    return success();
  }
};

struct MatmulOpLoweringToLinalg : public mlir::OpConversionPattern<demo::MatmulOp> {
  using mlir::OpConversionPattern<demo::MatmulOp>::OpConversionPattern;
  mlir::LogicalResult
  matchAndRewrite(demo::MatmulOp op, demo::MatmulOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    mlir::Location loc = op.getLoc();
    auto outputTy = op.getType().cast<mlir::ShapedType>();
    auto outputElementTy = outputTy.getElementType();
    auto zeroAttr = rewriter.getZeroAttr(outputElementTy);
    auto zero = rewriter.create<mlir::arith::ConstantOp>(loc, zeroAttr);
    auto initTensor = rewriter.create<mlir::linalg::InitTensorOp>(
        loc, outputTy.getShape(), outputTy.getElementType());
    mlir::Value zeroTensor =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);
    rewriter.replaceOpWithNewOp<mlir::linalg::MatmulOp>(
        op, TypeRange{op.getType()}, mlir::ValueRange{adaptor.lhs(), adaptor.rhs()},
        mlir::ValueRange{zeroTensor});
    return mlir::success();
  }
};

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(demo::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto adaptor = demo::TransposeOp::Adaptor(operands);
    rewriter.replaceOpWithNewOp<mlir::vector::TransposeOp>(
        op, adaptor.input(), mlir::ArrayRef<int64_t>{1, 0});
    return success();
  }
};

class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(demo::PrintOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("value = %f \n", 12), parentModule);

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<demo::PrintOp>(op);
    rewriter.create<CallOp>(
        loc, printfRef, rewriter.getIntegerType(32),
        ArrayRef<Value>({formatSpecifierCst, printOp.input()}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

class PrintVecOpLowering : public ConversionPattern {
public:
  explicit PrintVecOpLowering(MLIRContext *context)
      : ConversionPattern(demo::PrintVecOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto memRefType = (*op->operand_type_begin()).cast<ShapedType>();
    auto memRefShape = memRefType.getShape();
    auto loc = op->getLoc();

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();

    // Get a symbol reference to the printf function, inserting it if necessary.
    auto printfRef = getOrInsertPrintf(rewriter, parentModule);
    Value formatSpecifierCst = getOrCreateGlobalString(
        loc, rewriter, "frmt_spec", StringRef("%f \0", 4), parentModule);
    Value newLineCst = getOrCreateGlobalString(
        loc, rewriter, "nl", StringRef("\n\0", 2), parentModule);

    // Create a loop for each of the dimensions within the shape.
    SmallVector<Value, 4> loopIvs;
    for (unsigned i = 0, e = memRefShape.size(); i != e; ++i) {
      auto lowerBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
      auto upperBound = rewriter.create<mlir::arith::ConstantIndexOp>(loc, memRefShape[i]);
      auto step = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
      auto loop =
          rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step);
      for (Operation &nested : *loop.getBody())
        rewriter.eraseOp(&nested);
      loopIvs.push_back(loop.getInductionVar());

      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());

      // Insert a newline after each of the inner dimensions of the shape.
      if (i != e - 1)
        rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                                newLineCst);
      rewriter.create<scf::YieldOp>(loc);
      rewriter.setInsertionPointToStart(loop.getBody());
    }

    // Generate a call to printf for the current element of the loop.
    auto printOp = cast<demo::PrintVecOp>(op);
    auto elementLoad =
        rewriter.create<memref::LoadOp>(loc, printOp.input(), loopIvs);
    rewriter.create<CallOp>(loc, printfRef, rewriter.getIntegerType(32),
                            ArrayRef<Value>({formatSpecifierCst, elementLoad}));

    // Notify the rewriter that this operation has been removed.
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }

  /// Return a value representing an access into a global string with the given
  /// name, creating the string if necessary.
  static Value getOrCreateGlobalString(Location loc, OpBuilder &builder,
                                       StringRef name, StringRef value,
                                       ModuleOp module) {
    // Create the global at the entry of the module.
    LLVM::GlobalOp global;
    if (!(global = module.lookupSymbol<LLVM::GlobalOp>(name))) {
      OpBuilder::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto type = LLVM::LLVMArrayType::get(
          IntegerType::get(builder.getContext(), 8), value.size());
      global = builder.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true,
                                              LLVM::Linkage::Internal, name,
                                              builder.getStringAttr(value),
                                              /*alignment=*/0);
    }

    // Get the pointer to the first character in the global string.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, global);
    Value cst0 = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    return builder.create<LLVM::GEPOp>(
        loc,
        LLVM::LLVMPointerType::get(IntegerType::get(builder.getContext(), 8)),
        globalPtr, ArrayRef<Value>({cst0, cst0}));
  }
};

} // end anonymous namespace

namespace {

struct DemoToVectorLoweringPass
    : public PassWrapper<DemoToVectorLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<StandardOpsDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<FuncOp>();
    target.addLegalDialect<vector::VectorDialect>();
    target.addIllegalOp<demo::TransposeOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<TransposeOpLowering, MatmulOpLoweringToVector>(&getContext());

    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct DemoToLinalgLoweringPass
    : public mlir::PassWrapper<DemoToLinalgLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::StandardOpsDialect, mlir::arith::ArithmeticDialect, mlir::linalg::LinalgDialect>();
  }
  void runOnOperation() final {
    mlir::ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, FuncOp>();
    target.addLegalDialect<mlir::StandardOpsDialect, mlir::arith::ArithmeticDialect, mlir::linalg::LinalgDialect>();
    target.addIllegalDialect<demo::DemoDialect>();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpLoweringToLinalg>(&getContext());
    patterns.add<EmitDoubleNegPattern>(&getContext());
    auto module = getOperation();
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

struct DemoToLlvmLoweringPass
    : public PassWrapper<DemoToLlvmLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, StandardOpsDialect, vector::VectorDialect>();
  }
  void runOnOperation() final {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter typeConverter(&getContext());
    patterns.add<ScalarF64AddOpLowering, ScalarF64MulOpLowering>(&getContext());
    patterns.add<MatmulOpLoweringToLLVM>(typeConverter);
    patterns.add<PointWiseNegOpLowering>(typeConverter);
    patterns.add<PointWiseExpOpLowering>(typeConverter);
    patterns.add<PrintVecOpLowering>(&getContext());

    // mlir::FuncOp --> llvm::func
    populateStdToLLVMConversionPatterns(typeConverter, patterns);
    populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);


    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::demo::createLowerToVectorPass() {
  return std::make_unique<DemoToVectorLoweringPass>();
}

std::unique_ptr<mlir::Pass> mlir::demo::createLowerToLinalgPass() {
  return std::make_unique<DemoToLinalgLoweringPass>();
}

std::unique_ptr<mlir::Pass> mlir::demo::createLowerToLlvmPass() {
  return std::make_unique<DemoToLlvmLoweringPass>();
}
