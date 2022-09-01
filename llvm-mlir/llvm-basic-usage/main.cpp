#include <iostream>
#include <memory>
#include <stdarg.h>
#include <string>
#include <system_error>

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"

void printf_runtime(std::shared_ptr<llvm::IRBuilder<>> builder,
                    const char *format, ...) {
  llvm::Module *mod = builder->GetInsertBlock()->getModule();
  llvm::Function *func_printf = mod->getFunction("printf");
  if (!func_printf) {
    llvm::PointerType *Pty =
        llvm::PointerType::get(llvm::IntegerType::get(mod->getContext(), 8), 0);
    llvm::FunctionType *FuncTy9 = llvm::FunctionType::get(
        // why 32 bit...
        llvm::IntegerType::get(mod->getContext(), 32), true);

    func_printf = llvm::Function::Create(
        FuncTy9, llvm::GlobalValue::ExternalLinkage, "printf", mod);
    func_printf->setCallingConv(llvm::CallingConv::C);
  }

  llvm::Value *str = builder->CreateGlobalStringPtr(format);
  std::vector<llvm::Value *> call_params;
  call_params.push_back(str);

  va_list ap;
  va_start(ap, format);
  do {
    llvm::Value *op = va_arg(ap, llvm::Value *);
    if (op) {
      call_params.push_back(op);
    } else {
      break;
    }
  } while (1);
  va_end(ap);

  builder->CreateCall(func_printf, call_params);
}

template <typename T> void print(const std::string &title, T *v) {
  std::string s;
  llvm::raw_string_ostream rso(s);
  v->print(rso);
  std::cout << "========== " << title << " ==========" << std::endl;
  std::cout << s << std::endl;
};

void build_ir_fun(llvm::Module *m, llvm::LLVMContext *c) {
  /*
  build a function like this:

  void fun(int_8* x, float* y, int64_t size) {
    auto cx = reinterpret_cast<float*>(x);
    for (int64_t i = 0; i < size; ++i) {
      y[i] = cx[i] * 2;
    }
  }
  */
  auto *ir_fun = llvm::Function::Create(
      // Int8PtrTy <--> void*
      llvm::FunctionType::get(llvm::Type::getVoidTy(*c),
                              {llvm::Type::getInt8PtrTy(*c),
                               llvm::Type::getFloatPtrTy(*c),
                               llvm::Type::getInt64Ty(*c)},
                              false),
      llvm::Function::ExternalLinkage, "fun", *m);
  llvm::BasicBlock *init_block =
      llvm::BasicBlock::Create(*c, "init_block", ir_fun);
  auto init_builder = std::make_shared<llvm::IRBuilder<>>(init_block);
  printf_runtime(init_builder, "%s: size = %" PRId64 "\n",
                 init_builder->CreateGlobalStringPtr("At Runtime"),
                 static_cast<llvm::Value *>(ir_fun->arg_begin() + 2));
  llvm::Value *x = init_builder->CreateBitCast(
      ir_fun->arg_begin(), init_builder->getFloatTy()->getPointerTo());
  llvm::Value *counter_alloc = init_builder->CreateAlloca(
      init_builder->getInt64Ty(), nullptr, "counter");
  init_builder->CreateStore(init_builder->getInt64(0), counter_alloc);
  llvm::BasicBlock *cmp_block =
      llvm::BasicBlock::Create(*c, "cmp_block", ir_fun);
  init_builder->CreateBr(cmp_block);
  auto cmp_builder = std::make_shared<llvm::IRBuilder<>>(cmp_block);
  llvm::Value *counter = cmp_builder->CreateLoad(counter_alloc);
  llvm::Value *cond = cmp_builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_ULT,
                                             counter, ir_fun->arg_begin() + 2);
  llvm::BasicBlock *body_block =
      llvm::BasicBlock::Create(*c, "body_block", ir_fun);
  llvm::BasicBlock *end_block =
      llvm::BasicBlock::Create(*c, "end_block", ir_fun);
  cmp_builder->CreateCondBr(cond, body_block, end_block);
  auto body_builder = std::make_shared<llvm::IRBuilder<>>(body_block);
  llvm::Value *psrc = body_builder->CreateGEP(x, counter);
  llvm::Value *pdst = body_builder->CreateGEP(ir_fun->arg_begin() + 1, counter);
  body_builder->CreateStore(
      body_builder->CreateFMul(
          body_builder->CreateLoad(psrc),
          llvm::ConstantFP::get(llvm::Type::getFloatTy(*c), 2.0)),
      pdst);
  body_builder->CreateStore(
      body_builder->CreateAdd(counter, body_builder->getInt64(1)),
      counter_alloc);
  body_builder->CreateBr(cmp_block);
  auto end_builder = std::make_shared<llvm::IRBuilder<>>(end_block);
  end_builder->CreateStore(end_builder->getInt64(0), counter_alloc);
  end_builder->CreateRetVoid();
}

int main() {
  // necessary: initialize llvm: target machine ...
  struct Guard {
    Guard() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();
    }
  };
  static Guard g;

  // necessary: initialize: context && module; must be unique_prt...
  auto ir_context = std::make_unique<llvm::LLVMContext>();
  auto ir_module = std::make_unique<llvm::Module>("m", *ir_context);

  // optional: set triple && datalayout
  // m.ll: target datalayout
  ir_module->setDataLayout(
      "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128");
  // m.ll: target triple
  ir_module->setTargetTriple("x86_64-unknown-linux-gnu");

  build_ir_fun(ir_module.get(), ir_context.get());

  // save module ir to file
  std::error_code ec;
  llvm::raw_fd_ostream rfo("m.ll", ec);
  ir_module->print(rfo, nullptr);

  // debug ir ... compile infomation
  // support: llvm::Value, llvm::Type ...
  auto ir_fun = ir_module->getFunction("fun");
  print("ir_fun", ir_fun);
  print("ir_fun->getType()", ir_fun->getType());

  // compile ir_fun
  llvm::verifyModule(*ir_module, &llvm::outs());
  llvm::orc::ThreadSafeModule tsm(std::move(ir_module), std::move(ir_context));
  llvm::ExitOnError err;
  auto jit = err(llvm::orc::LLJITBuilder().create());
  err(jit->addIRModule(std::move(tsm)));
  auto generator =
      err(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix()));
  jit->getMainJITDylib().addGenerator(std::move(generator));
  using FunctionType = void (*)(void *, float *, int64_t);
  auto fun =
      reinterpret_cast<FunctionType>(err(jit->lookup("fun")).getAddress());
  // run ir_fun: just like a normal function
  float src[] = {1.0, 2.0};
  float dst[] = {0.0, 0.0};
  fun(src, dst, 2);
  std::cout << dst[0] << ' ' << dst[1] << std::endl;

  return 0;
}