#include <stdarg.h>
#include <iostream>
#include <memory>
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

const char* func_name = "llvm_entry_func";

template <typename... Args>
void insert_printf(std::shared_ptr<llvm::IRBuilder<>> builder, const char* format, Args... args) {
  llvm::Module* llvm_module = builder->GetInsertBlock()->getModule();
  llvm::Function* printf_func = llvm_module->getFunction("printf");
  if (!printf_func) {
    auto i8_ptr_type = builder->getInt8PtrTy();
    auto i32_type = builder->getInt32Ty();

    auto printf_func_type = llvm::FunctionType::get(i32_type, llvm::ArrayRef<llvm::Type*>({i8_ptr_type}), true);
    printf_func = llvm::Function::Create(printf_func_type, llvm::GlobalValue::ExternalLinkage, "printf", llvm_module);
    printf_func->setCallingConv(llvm::CallingConv::C);
  }

  std::vector<llvm::Value*> args_v({builder->CreateGlobalStringPtr(format), (nullptr, args)...});
  builder->CreateCall(printf_func, args_v);
}

void build_ir_fun(llvm::Module* m, llvm::LLVMContext* c) {
  /*

  void fun(int_8* src, float* dst, int64_t size) {
    printf("%s: size = %li\n", "At Runtime", size);
    auto src = reinterpret_cast<float*>(src);
    for (int64_t i = 0; i < size; ++i) {
      dst[i] = src[i] * 2;
    }
  }

  */
  auto builder = std::make_shared<llvm::IRBuilder<>>(*c);
  llvm::Type* void_type = builder->getVoidTy();
  llvm::Type* i64_type = builder->getInt64Ty();
  llvm::Type* f32_type = builder->getFloatTy();
  llvm::Type* i8_ptr_type = builder->getInt8PtrTy();
  llvm::Type* f32_ptr_type = llvm::Type::getFloatPtrTy(*c);

  auto func_type = llvm::FunctionType::get(void_type, {i8_ptr_type, f32_ptr_type, i64_type}, false);
  auto ir_fun = llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, func_name, *m);

  auto init_block = llvm::BasicBlock::Create(*c, "init_block", ir_fun);
  auto init_builder = std::make_shared<llvm::IRBuilder<>>(init_block);
  auto cmp_block = llvm::BasicBlock::Create(*c, "cmp_block", ir_fun);
  auto cmp_builder = std::make_shared<llvm::IRBuilder<>>(cmp_block);
  auto body_block = llvm::BasicBlock::Create(*c, "body_block", ir_fun);
  auto body_builder = std::make_shared<llvm::IRBuilder<>>(body_block);
  auto end_block = llvm::BasicBlock::Create(*c, "end_block", ir_fun);
  auto end_builder = std::make_shared<llvm::IRBuilder<>>(end_block);

  llvm::Value* src_f32_ptr = init_builder->CreateBitCast(ir_fun->getArg(0), f32_ptr_type);
  llvm::Value* dst_f32_ptr = ir_fun->getArg(1);
  llvm::Value* size_i64 = ir_fun->getArg(2);

  insert_printf(init_builder, "%s: size = %li\n", init_builder->CreateGlobalStringPtr("At Runtime"), size_i64);
  llvm::Value* cnt_ptr = init_builder->CreateAlloca(i64_type);
  init_builder->CreateStore(init_builder->getInt64(0), cnt_ptr);
  init_builder->CreateBr(cmp_block);

  llvm::Value* cnt = cmp_builder->CreateLoad(i64_type, cnt_ptr);
  llvm::Value* cond = cmp_builder->CreateCmp(llvm::CmpInst::Predicate::ICMP_ULT, cnt, size_i64);
  cmp_builder->CreateCondBr(cond, body_block, end_block);

  insert_printf(body_builder, "cnt = %li\n", cnt);
  llvm::Value* psrc = body_builder->CreateGEP(f32_type, src_f32_ptr, cnt);
  insert_printf(body_builder, "src = %p\n", psrc);
  llvm::Value* pdst = body_builder->CreateGEP(f32_type, dst_f32_ptr, cnt);
  insert_printf(body_builder, "dst = %p\n", pdst);
  llvm::Value* cst_f32_2 = llvm::ConstantFP::get(f32_type, 2.0);
  body_builder->CreateStore(body_builder->CreateFMul(body_builder->CreateLoad(f32_type, psrc), cst_f32_2), pdst);
  body_builder->CreateStore(body_builder->CreateAdd(cnt, body_builder->getInt64(1)), cnt_ptr);
  body_builder->CreateBr(cmp_block);

  end_builder->CreateRetVoid();
}

int main() {
  struct Guard {
    Guard() {
      llvm::InitializeNativeTarget();
      llvm::InitializeNativeTargetAsmPrinter();
      llvm::InitializeNativeTargetAsmParser();
    }
  };
  Guard g;

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = std::make_unique<llvm::Module>("llvm_module", *llvm_context);

  build_ir_fun(llvm_module.get(), llvm_context.get());

  llvm::verifyModule(*llvm_module, &llvm::outs());
  llvm::orc::ThreadSafeModule tsm(std::move(llvm_module), std::move(llvm_context));
  llvm::ExitOnError err;
  auto jit = err(llvm::orc::LLJITBuilder().create());
  err(jit->addIRModule(std::move(tsm)));
  auto generator =
      err(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(jit->getDataLayout().getGlobalPrefix()));
  jit->getMainJITDylib().addGenerator(std::move(generator));
  using FunctionType = void (*)(void*, float*, int64_t);
  auto func = err(jit->lookup(func_name)).toPtr<FunctionType>();

  std::vector<float> src({1.0, 2.0});
  std::vector<float> dst({0.0, 0.0});
  func(static_cast<void*>(src.data()), dst.data(), 2);
  for (auto v : dst) {
    std::cout << v << ' ';
  }
  std::cout << std::endl;

  return 0;
}