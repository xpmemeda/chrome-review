#include <cassert>
#include <exception>
#include <memory>
#include <vector>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

void register_dialects(mlir::MLIRContext& context) {
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::AffineDialect>();
  context.loadDialect<mlir::arith::ArithmeticDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
}

void lower_to_llvm_dialect(mlir::ModuleOp& module_op);
void insert_printf_i64(mlir::Value value, mlir::OpBuilder& builder, mlir::ModuleOp module_op, mlir::Location loc);
std::unique_ptr<llvm::Module> convert_to_llvm_ir(mlir::ModuleOp module, llvm::LLVMContext* llvm_context);
int write_llvm_bitcode_to_file(std::unique_ptr<llvm::Module> llvm_module, std::string file_path);

int create_memref_alloc_with_affine_map(mlir::OpBuilder& builder, mlir::Location loc, mlir::ModuleOp module_op) {
  mlir::AffineMap map;
  {
    std::vector<mlir::AffineExpr> exprs({builder.getAffineDimExpr(0) % 2, builder.getAffineDimExpr(1) % 3});
    map = mlir::AffineMap::get(2, 0, exprs, builder.getContext());
  }
  auto memref_type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({4, 9}), builder.getI64Type(), map);
  auto alloc = builder.create<mlir::memref::AllocOp>(loc, memref_type);
  {
    mlir::Value v = builder.create<mlir::arith::ConstantIntOp>(loc, 29, builder.getI64Type());
    mlir::Value cst_2 = builder.create<mlir::arith::ConstantIndexOp>(loc, 2);
    mlir::Value cst_3 = builder.create<mlir::arith::ConstantIndexOp>(loc, 3);
    builder.create<mlir::AffineStoreOp>(loc, v, alloc, mlir::ValueRange({cst_2, cst_3}));
  }
  {
    mlir::Value cst_0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value v = builder.create<mlir::AffineLoadOp>(loc, alloc, mlir::ValueRange({cst_0, cst_0}));
    insert_printf_i64(v, builder, module_op, loc);
  }
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({}));
  llvm::errs() << module_op << '\n';
  // NOTE.
  // normalizeMemRef会把所有AffineMap变成Identity的形式，并且修改后面相关的加载算子
  // 但是只能纠正AffineLoadOp，不能纠正memref::LoadOp，也许是因为AffineMap只是Affine Dialect的内容
  // 带有非Identity AffineMap的MemRef不能降级到LLVM
  assert(mlir::succeeded(mlir::normalizeMemRef(&alloc)) && "normalize memref err");
  llvm::errs() << module_op << '\n';
  lower_to_llvm_dialect(module_op);
  llvm::errs() << module_op << '\n';

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = convert_to_llvm_ir(module_op, llvm_context.get());
  write_llvm_bitcode_to_file(std::move(llvm_module), "llvm_module");
  return 0;
}

int create_memref_broadcast_store(mlir::OpBuilder& builder, mlir::Location loc, mlir::ModuleOp module_op) {
  auto alloc_type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({1, 3}), builder.getI64Type());
  auto alloc = builder.create<mlir::memref::AllocaOp>(loc, alloc_type);
  mlir::AffineExpr expr = builder.getAffineSymbolExpr(0) +
                          builder.getAffineDimExpr(0) * builder.getAffineSymbolExpr(1) +
                          builder.getAffineDimExpr(1) * builder.getAffineSymbolExpr(2);
  mlir::AffineMap map = mlir::AffineMap::get(2, 3, expr, builder.getContext());
  auto r_cast_type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({3, 3}), builder.getI64Type(), map);
  mlir::OpFoldResult r_cast_offset(builder.getIndexAttr(0));
  std::vector<mlir::OpFoldResult> r_cast_sizes({builder.getIndexAttr(3), builder.getIndexAttr(3)});
  std::vector<mlir::OpFoldResult> r_cast_strides({builder.getIndexAttr(0), builder.getIndexAttr(1)});
  auto r_cast = builder.create<mlir::memref::ReinterpretCastOp>(
      loc, r_cast_type, alloc, r_cast_offset, r_cast_sizes, r_cast_strides);

  auto loop_call_back = [&](mlir::OpBuilder& builder, mlir::Location loc, mlir::ValueRange ivs) {
    mlir::Value cst_3 = builder.create<mlir::arith::ConstantIndexOp>(loc, 3);
    mlir::Value v = builder.create<mlir::arith::MulIOp>(loc, ivs[0], cst_3);
    v = builder.create<mlir::arith::AddIOp>(loc, v, ivs[1]);
    v = builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(), v);
    builder.create<mlir::AffineStoreOp>(loc, v, r_cast, ivs);
  };
  mlir::Value cst_0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value cst_3 = builder.create<mlir::arith::ConstantIndexOp>(loc, 3);
  std::vector<mlir::Value> lbs(2, cst_0);
  std::vector<mlir::Value> ubs(2, cst_3);
  std::vector<int64_t> steps(2, 1);
  mlir::buildAffineLoopNest(builder, loc, lbs, ubs, steps, loop_call_back);

  mlir::Value v = builder.create<mlir::memref::LoadOp>(loc, alloc, mlir::ValueRange({cst_0, cst_0}));
  insert_printf_i64(v, builder, module_op, loc);

  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({}));

  llvm::errs() << module_op << '\n';
  lower_to_llvm_dialect(module_op);
  llvm::errs() << module_op << '\n';

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = convert_to_llvm_ir(module_op, llvm_context.get());
  write_llvm_bitcode_to_file(std::move(llvm_module), "llvm_module");

  return 0;
}

int create_reinterpret_cast_memref_to_zero_stride(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::ModuleOp module_op) {
  auto alloc_type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({2, 1}), builder.getI64Type());
  auto alloc = builder.create<mlir::memref::AllocaOp>(loc, alloc_type);
  {
    mlir::Value v = builder.create<mlir::arith::ConstantIntOp>(loc, 29, builder.getI64Type());
    mlir::Value cst_0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    builder.create<mlir::memref::StoreOp>(loc, v, alloc, mlir::ValueRange({cst_0, cst_0}));
  }

  // NOTE.
  // MemRefType会自动完成以下推导：
  //     1) 它的offset为零
  //     2) 它会根据静态维度来自动计算静态stride，只有遇到动态stride才会去内存中读取
  // 所以说，如果我们要用ReinterpretcastOp来显式设置offset和stride，必须告诉MemRefType哪些值要从内存中读取
  //     1) 目前只能通过设置AffineMap来完成这项工作
  //     2) SymbolExpr(0)是offset，后面的是各维度的stride
  // 一个示例：
  mlir::AffineMap map;
  {
    // 这里前面的``builder.getAffineSymbolExpr(0)``可以不要，在这个示例之下，我们并没有改变offset
    mlir::AffineExpr expr = builder.getAffineSymbolExpr(0) +
                            builder.getAffineDimExpr(0) * builder.getAffineSymbolExpr(1) +
                            builder.getAffineDimExpr(1) * builder.getAffineSymbolExpr(2);
    map = mlir::AffineMap::get(2, 3, expr, builder.getContext());
  }
  auto r_cast_type = mlir::MemRefType::get(llvm::ArrayRef<int64_t>({2, 3}), builder.getI64Type(), map);
  mlir::OpFoldResult r_cast_offset(builder.getIndexAttr(0));
  std::vector<mlir::OpFoldResult> r_cast_sizes({builder.getIndexAttr(2), builder.getIndexAttr(3)});
  std::vector<mlir::OpFoldResult> r_cast_strides({builder.getIndexAttr(1), builder.getIndexAttr(0)});
  auto r_cast = builder.create<mlir::memref::ReinterpretCastOp>(
      loc, r_cast_type, alloc, r_cast_offset, r_cast_sizes, r_cast_strides);

  mlir::Value cst_0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value cst_1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
  mlir::Value v = builder.create<mlir::memref::LoadOp>(loc, r_cast, mlir::ValueRange({cst_0, cst_1}));
  insert_printf_i64(v, builder, module_op, loc);
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange({}));

  llvm::errs() << module_op << '\n';
  lower_to_llvm_dialect(module_op);
  llvm::errs() << module_op << '\n';

  auto llvm_context = std::make_unique<llvm::LLVMContext>();
  auto llvm_module = convert_to_llvm_ir(module_op, llvm_context.get());
  write_llvm_bitcode_to_file(std::move(llvm_module), "llvm_module");

  return 0;
}

int main() {
  mlir::MLIRContext context;
  register_dialects(context);
  mlir::OpBuilder builder(&context);

  auto loc = builder.getUnknownLoc();

  auto module_op = mlir::ModuleOp::create(loc);
  auto func_op = mlir::func::FuncOp::create(loc, "mlir_entry_func", builder.getFunctionType({}, llvm::None));
  module_op.push_back(func_op);
  builder.setInsertionPointToStart(func_op.addEntryBlock());

  return create_reinterpret_cast_memref_to_zero_stride(builder, loc, module_op);
  // return create_memref_broadcast_store(builder, loc, module_op);
}
