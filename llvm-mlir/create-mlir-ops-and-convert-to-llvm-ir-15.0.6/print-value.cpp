#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

mlir::FlatSymbolRefAttr get_or_insert_printf(mlir::OpBuilder& builder, mlir::ModuleOp module_op) {
  auto* context = module_op.getContext();
  if (module_op.lookupSymbol<mlir::LLVM::LLVMFuncOp>("printf")) return mlir::SymbolRefAttr::get(context, "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  auto llvmI32Ty = mlir::IntegerType::get(context, 32);
  auto llvmI8PtrTy = mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(context, 8));
  auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
      /*isVarArg=*/true);

  mlir::OpBuilder::InsertionGuard insert_guard(builder);
  builder.setInsertionPointToStart(module_op.getBody());
  builder.create<mlir::LLVM::LLVMFuncOp>(module_op.getLoc(), "printf", llvmFnType);
  return mlir::SymbolRefAttr::get(context, "printf");
}

mlir::Value get_or_create_global_string(
    mlir::Location loc, mlir::OpBuilder& builder, mlir::StringRef name, mlir::StringRef value, mlir::ModuleOp module) {
  // Create the global at the entry of the module.
  mlir::LLVM::GlobalOp global;
  if (!(global = module.lookupSymbol<mlir::LLVM::GlobalOp>(name))) {
    mlir::OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto type = mlir::LLVM::LLVMArrayType::get(mlir::IntegerType::get(builder.getContext(), 8), value.size());
    global = builder.create<mlir::LLVM::GlobalOp>(loc, type, /*isConstant=*/true, mlir::LLVM::Linkage::Internal, name,
        builder.getStringAttr(value),
        /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  mlir::Value globalPtr = builder.create<mlir::LLVM::AddressOfOp>(loc, global);
  mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
      loc, mlir::IntegerType::get(builder.getContext(), 64), builder.getIntegerAttr(builder.getIndexType(), 0));
  return builder.create<mlir::LLVM::GEPOp>(loc,
      mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(builder.getContext(), 8)), globalPtr,
      llvm::ArrayRef<mlir::Value>({cst0, cst0}));
}

void insert_printf_i64(mlir::Value value, mlir::OpBuilder& builder, mlir::ModuleOp module_op, mlir::Location loc) {
  auto printf_ref = get_or_insert_printf(builder, module_op);
  auto fmt_string =
      get_or_create_global_string(loc, builder, "printf_fmt_i64", llvm::StringRef("int64: %d \n\0", 12), module_op);
  builder.create<mlir::func::CallOp>(
      loc, printf_ref, builder.getI32Type(), llvm::ArrayRef<mlir::Value>({fmt_string, value}));
  return;
}