#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "dialect.h"
#include "passes.h"

using namespace mlir;
using namespace mlir::tfcc;

// namespace {
// class AddOpLowering : public ConversionPattern {
// public:
//     AddOpLowering(MLIRContext *ctx)
//         : ConversionPattern(tfcc::AddOp::getOperationName(), 1, ctx) {}
//     LogicalResult
//     matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                     ConversionPatternRewriter &rewriter) const override {
//         return LLVM::detail::vectorOneToOneRewrite(
//             op, LLVM::FAddOp::getOperationName(), operands, 
//         )
//     }
// private:
// };

// class MulOpLowering : public ConvertOpToLLVMPattern<LLVM::FMulOp> {
// public:
//     // Ctor
//     using ConvertOpToLLVMPattern<LLVM::FMulOp>::ConvertOpToLLVMPattern;
//     LogicalResult
//     matchAndRewrite(Operation *op, ArrayRef<Value> operands,
//                     ConversionPatternRewriter &rewriter) const override {
//         return LLVM::detail::vector
//     }
// };

// class ConstantOpLowering;
// }
namespace {
using AddOpLowering = VectorConvertToLLVMPattern<AddOp, LLVM::FAddOp>;
using MulOpLowering = VectorConvertToLLVMPattern<MulOp, LLVM::FMulOp>;
struct ConstantOpLowering : public ConvertOpToLLVMPattern<ConstantOp> {
    using ConvertOpToLLVMPattern<ConstantOp>::ConvertOpToLLVMPattern;
    LogicalResult
    matchAndRewrite(ConstantOp op, ArrayRef<Value> operands,
                    ConversionPatternRewriter &rewriter) const override {
        return LLVM::detail::oneToOneRewrite(
            op, LLVM::ConstantOp::getOperationName(), operands, *getTypeConverter(),
            rewriter);
    }
};
}

namespace {
struct TfccToLlvmLoweringPass
    : public PassWrapper<TfccToLlvmLoweringPass, OperationPass<ModuleOp>> {
    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
    }
    void runOnOperation() final;
};
}

void TfccToLlvmLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());
    target.addLegalOp<ModuleOp>();

    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter typeConverter(&getContext());
    patterns.add<AddOpLowering, MulOpLowering, ConstantOpLowering>(typeConverter);

    auto module = getOperation();
    if (failed(applyFullConversion(module, target, std::move(patterns))))
        signalPassFailure();
}

std::unique_ptr<mlir::Pass> mlir::tfcc::createLowerToLlvmPass() {
    return std::make_unique<TfccToLlvmLoweringPass>();
}
