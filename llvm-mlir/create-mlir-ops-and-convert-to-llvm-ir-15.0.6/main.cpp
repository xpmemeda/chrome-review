#include <cassert>
#include <exception>
#include <memory>
#include <vector>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

void lower_to_llvm_dialect(mlir::ModuleOp& module_op) {
  std::unique_ptr<mlir::Pass> create_lower_to_llvm_pass();
  mlir::PassManager pm(module_op.getContext());
  pm.addPass(create_lower_to_llvm_pass());
  assert(mlir::succeeded(pm.run(module_op)) && "lower to llvm err");
}

mlir::AffineMap create_affine_map_1(mlir::OpBuilder& builder) {
  // NOTE: semi-affine expressions (modulo by non-const) are not supported
  mlir::AffineExpr expr = builder.getAffineDimExpr(0) % 2 * 3 + builder.getAffineDimExpr(1) % 3;
  return mlir::AffineMap::get(2, 2, expr, builder.getContext());
}

mlir::Value insert_memref_alloc(
    mlir::MemRefType type, mlir::ValueRange dyn_dims, mlir::Location loc, mlir::OpBuilder& builder) {
  mlir::IntegerAttr alignment_attr = builder.getI64IntegerAttr(16);
  if (type.getShape().size() == 0) {
    return builder.create<mlir::memref::AllocOp>(loc, type);
  } else {
    return builder.create<mlir::memref::AllocOp>(loc, type, dyn_dims, alignment_attr);
  }
}

void register_dialects(mlir::MLIRContext& context) {
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::AffineDialect>();
  context.loadDialect<mlir::arith::ArithmeticDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
}

void insert_printf_i64(mlir::Value value, mlir::OpBuilder& builder, mlir::ModuleOp module_op, mlir::Location loc);
std::unique_ptr<llvm::Module> convert_to_llvm_ir(mlir::ModuleOp module, llvm::LLVMContext* llvm_context);
int write_llvm_bitcode_to_file(std::unique_ptr<llvm::Module> llvm_module, std::string file_path);

int main() {
  mlir::MLIRContext context;
  register_dialects(context);
  mlir::OpBuilder builder(&context);

  auto loc = builder.getUnknownLoc();

  auto module_op = mlir::ModuleOp::create(loc);
  auto func_op = mlir::func::FuncOp::create(loc, "mlir_entry_func", builder.getFunctionType({}, llvm::None));
  module_op.push_back(func_op);

  builder.setInsertionPointToStart(func_op.addEntryBlock());
  auto d0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 10);
  auto d1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 10);
  auto s0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
  auto s1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 3);
  mlir::Value r =
      builder.create<mlir::AffineApplyOp>(loc, create_affine_map_1(builder), mlir::ValueRange({d0, d1, s0, s1}));
  r = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), r);
  insert_printf_i64(r, builder, module_op, loc);
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({}));
  llvm::errs() << module_op << '\n';
  lower_to_llvm_dialect(module_op);
  llvm::errs() << module_op << '\n';

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = convert_to_llvm_ir(module_op, llvm_context.get());
  write_llvm_bitcode_to_file(std::move(llvm_module), "llvm_module");
  return 0;
}