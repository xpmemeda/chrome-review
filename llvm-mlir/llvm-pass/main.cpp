#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

std::string load_data_from_file(const std::string &path) {
  using std::ios;
  std::ifstream ifs(path, ios::binary);
  if (!ifs) {
    throw std::runtime_error("open file: \"" + path + "\" failed");
  }
  std::string data(std::istreambuf_iterator<char>(ifs), {});
  return data;
}

std::tuple<std::unique_ptr<llvm::LLVMContext>, std::unique_ptr<llvm::Module>>
getModule() {
  auto c = std::make_unique<llvm::LLVMContext>();
  auto memoryBuffer =
      llvm::MemoryBuffer::getMemBufferCopy(load_data_from_file("../ir.ll"));
  llvm::SMDiagnostic err;
  auto m = llvm::parseIR(*memoryBuffer.get(), err, *c);
  return std::make_tuple(std::move(c), std::move(m));
}

void run_fun(std::unique_ptr<llvm::Module> m,
             std::unique_ptr<llvm::LLVMContext> c, const std::string &fname) {
  // compile ir_fun
  llvm::verifyModule(*m, &llvm::outs());
  llvm::orc::ThreadSafeModule tsm(std::move(m), std::move(c));
  llvm::ExitOnError err;
  auto jit = err(llvm::orc::LLJITBuilder().create());
  err(jit->addIRModule(std::move(tsm)));
  auto generator =
      err(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix()));
  jit->getMainJITDylib().addGenerator(std::move(generator));
  using FunctionType = void (*)(void *, float *, int64_t);
  auto fun =
      reinterpret_cast<FunctionType>(err(jit->lookup(fname)).getAddress());
  // run ir_fun: just like a normal function
  float src[] = {1.0, 2.0};
  float dst[] = {0.0, 0.0};
  fun(src, dst, 2);
  std::cout << dst[0] << ' ' << dst[1] << std::endl;
}

int main() {
  struct Guard {
    Guard() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();
    }
  };
  static Guard g;

  std::unique_ptr<llvm::LLVMContext> c;
  std::unique_ptr<llvm::Module> m;

  std::tie(c, m) = getModule();

  run_fun(std::move(m), std::move(c), "fun");

  return 0;
}