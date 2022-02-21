#include <stdexcept>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "dialect.h"
#include "passes.h"

using namespace mlir::tfcc;

mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context) {
    mlir::OpBuilder builder(&context);
    auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto func_type = builder.getFunctionType(llvm::None, llvm::None);
    auto func = mlir::FuncOp::create(builder.getUnknownLoc(), "func", func_type);
    if (!func) {
        printf("create func error.\n");
        return theModule;
    }

    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value a = builder.create<ConstantOp>(builder.getUnknownLoc(), 0.1);
    mlir::Value b = builder.create<ConstantOp>(builder.getUnknownLoc(), 1.0);
    mlir::Value c = builder.create<ConstantOp>(builder.getUnknownLoc(), 10.);
    mlir::Value d = builder.create<AddOp>(builder.getUnknownLoc(), a, b);
    mlir::Value e = builder.create<MulOp>(builder.getUnknownLoc(), c, d);

    theModule.push_back(func);

    return theModule;
}

void mlirOpt(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::tfcc::createLowerToLlvmPass());
    if (mlir::failed(pm.run(*module))) {
        printf("pm run error.\n");
    }
}

int main(int argc, char **argv) {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tfcc::TfccDialect>();
    mlir::OwningModuleRef module = mlirGen(context);
    module->dump();
    mlirOpt(context, module);
    module->dump();
    return 0;
}
