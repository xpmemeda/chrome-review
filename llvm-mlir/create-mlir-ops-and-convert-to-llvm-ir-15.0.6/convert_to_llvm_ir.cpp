#include <fstream>
#include <memory>

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

std::unique_ptr<llvm::Module> convert_to_llvm_ir(mlir::ModuleOp module, llvm::LLVMContext* llvm_context) {
  // Register the translation to LLVM IR with the MLIR context.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // Convert the module to LLVM IR in a new LLVM IR context.
  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvm_context);
  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return nullptr;
  }

  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::ExecutionEngine::setupTargetTriple(llvmModule.get());

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return nullptr;
  }
  llvm::errs() << *llvmModule << "\n";
  return llvmModule;
}

int write_llvm_bitcode_to_file(std::unique_ptr<llvm::Module> llvm_module, std::string file_path) {
  std::string data;
  llvm::raw_string_ostream os(data);
  llvm::WriteBitcodeToFile(*llvm_module, os);
  std::ofstream ofs(file_path);
  ofs << data;
  return 0;
}