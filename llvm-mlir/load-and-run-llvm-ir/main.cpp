#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include <fstream>
#include <iostream>

std::string load_data_from_file(const std::string &path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error("failed to load ir");
  }
  return {std::istreambuf_iterator<char>(ifs), {}};
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    std::cout << "Usage: " << argv[0] << " <llvm-ir-path>" << std::endl;
    return 1;
  }

  std::string data = load_data_from_file(argv[1]);

  struct Guard {
    Guard() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();
    }
  };
  static Guard g;

  auto context = std::make_unique<llvm::LLVMContext>();
  auto memory_buffer = llvm::MemoryBuffer::getMemBuffer(data);

  llvm::ExitOnError err;
  auto m = err(llvm::parseBitcodeFile(*memory_buffer, *context));

  llvm::verifyModule(*m, &llvm::outs());
  llvm::orc::ThreadSafeModule tsm(std::move(m), std::move(context));
  auto jit = err(llvm::orc::LLJITBuilder().create());
  err(jit->addIRModule(std::move(tsm)));
  auto generator =
      err(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix()));
  jit->getMainJITDylib().addGenerator(std::move(generator));
  using FunctionType = void (*)();
  auto fun =
      reinterpret_cast<FunctionType>(err(jit->lookup("func")).getAddress());
  fun();

  return 0;
}
