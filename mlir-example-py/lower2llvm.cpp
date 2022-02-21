#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "dialect.h"
#include "passes.h"

using namespace mlir;

namespace {
// using AddOpLowering = VectorConvertToLLVMPattern<tfcc::AddOp, LLVM::FAddOp>;
struct AddOpLowering : public ConversionPattern {
  AddOpLowering(MLIRContext *ctx)
      : ConversionPattern(tfcc::AddOp::getOperationName(), 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::errs() << "AddOpLowering:\n";
    llvm::errs() << operands.size() << "\n";
    llvm::errs() << "\t" << *operands.begin() << "\n";
    llvm::errs() << "\t" << *(operands.begin() + 1) << "\n";
    rewriter.replaceOpWithNewOp<LLVM::FAddOp>(op, operands);
    return success();
  }
};
// using MulOpLowering = VectorConvertToLLVMPattern<tfcc::MulOp, LLVM::FMulOp>;
// struct MulOpLowering : public ConvertOpToLLVMPattern<tfcc::MulOp> {
//   using ConvertOpToLLVMPattern<tfcc::MulOp>::ConvertOpToLLVMPattern;
//   LogicalResult
//   matchAndRewrite(tfcc::MulOp op, ArrayRef<Value> operands,
//                   ConversionPatternRewriter &rewriter) const override {
//     llvm::errs() << "MulOpLowering:\n";
//     rewriter.replaceOpWithNewOp<LLVM::MulOp>(op, operands);
//     return success();
//   }
// };
struct MulOpLowering : public ConversionPattern {
  MulOpLowering(MLIRContext *ctx)
      : ConversionPattern(tfcc::MulOp::getOperationName(), 1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    llvm::errs() << "MulOpLowering:\n";
    llvm::errs() << operands.size() << "\n";
    llvm::errs() << "\t" << *operands.begin() << "\n";
    llvm::errs() << "\t" << *(operands.begin() + 1) << "\n";
    rewriter.replaceOpWithNewOp<LLVM::FMulOp>(op, operands);
    return success();
  }
};
// The pattern will not be executed if ``tfcc::ConstantOp`` has been added as
// legal
struct ConstantOpLowering : public ConvertOpToLLVMPattern<tfcc::ConstantOp> {
  using ConvertOpToLLVMPattern<tfcc::ConstantOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(tfcc::ConstantOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::errs() << "ConstantOpLowering:\n";
    llvm::errs() << operands.size() << "\n"; // 0
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto *ctx = module.getContext();
    rewriter.replaceOpWithNewOp<LLVM::ConstantOp>(op, Float64Type::get(ctx),
                                                  op.valueAttr());
    return success();
    // llvm::errs() will print out messages even if the pattern failure.
    // return failure();
  }
};
struct ReturnOpLowering : public ConvertOpToLLVMPattern<tfcc::ReturnOp> {
  using ConvertOpToLLVMPattern<tfcc::ReturnOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(tfcc::ReturnOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ReturnOp>(op, TypeRange(), ValueRange(),
                                                op->getAttrs());
    return success();
  }
};
class PrintOpLowering : public ConversionPattern {
public:
  explicit PrintOpLowering(MLIRContext *context)
      : ConversionPattern(tfcc::PrintOp::getOperationName(), 1, context) {}

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
    auto printOp = cast<tfcc::PrintOp>(op);
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
} // namespace

namespace {
struct TfccToLlvmLoweringPass
    : public PassWrapper<TfccToLlvmLoweringPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect, StandardOpsDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void TfccToLlvmLoweringPass::runOnOperation() {
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter typeConverter(&getContext());
  patterns.add<AddOpLowering, MulOpLowering>(&getContext());
  patterns.add<ConstantOpLowering, ReturnOpLowering>(typeConverter);
  patterns.add<PrintOpLowering>(&getContext());

  // mlir::FuncOp --> llvm::func
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::tfcc::createLowerToLlvmPass() {
  return std::make_unique<TfccToLlvmLoweringPass>();
}
