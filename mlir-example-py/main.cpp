#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "dialect.h"
#include "passes.h"

using namespace mlir::tfcc;

mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context) {
    mlir::OpBuilder builder(&context);
    auto theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    auto func_type = builder.getFunctionType(llvm::None, llvm::None);
    auto func = mlir::FuncOp::create(builder.getUnknownLoc(), "main", func_type);
    if (!func) {
        llvm::errs() << "Failed to create MLIR FuncOp\n";
        return theModule;
    }

    auto entryBlock = func.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::Value a = builder.create<ConstantOp>(builder.getUnknownLoc(), 0.1);
    mlir::Value b = builder.create<ConstantOp>(builder.getUnknownLoc(), 1.0);
    mlir::Value c = builder.create<ConstantOp>(builder.getUnknownLoc(), 10.);
    mlir::Value d = builder.create<AddOp>(builder.getUnknownLoc(), a, b);
    mlir::Value e = builder.create<MulOp>(builder.getUnknownLoc(), c, d);
    builder.create<PrintOp>(builder.getUnknownLoc(), e);
    builder.create<ReturnOp>(builder.getUnknownLoc());
    llvm::outs() << e << "\n";

    theModule.push_back(func);

    theModule.dump();

    if (mlir::failed(mlir::verify(theModule))) {
        theModule.emitError("module verification error");
        return nullptr;
    }

    return theModule;
}

void mlirOpt(mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
    mlir::PassManager pm(&context);
    pm.addPass(mlir::tfcc::createLowerToLlvmPass());
    if (mlir::failed(pm.run(*module))) {
        llvm::errs() << "Failed to lower dialect\n";
    }
}

std::unique_ptr<llvm::Module> mlirToLlvm(mlir::ModuleOp module) {
    mlir::registerLLVMDialectTranslation(*module->getContext());
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return nullptr;
    }
    llvm::errs() << *llvmModule << "\n";
    // llvm::InitializeNativeTarget();
    // llvm::InitializeNativeTargetAsmPrinter();
    return nullptr;
}

int runJit(mlir::ModuleOp module) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  bool enableOpt = false;
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}

int func_main() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tfcc::TfccDialect>();
    mlir::OwningModuleRef module = mlirGen(context);
    if (!module) {
        return 1;
    }
    module->dump();
    mlirOpt(context, module);
    module->dump();
    mlirToLlvm(*module);
    runJit(*module);
    return 0;
}
