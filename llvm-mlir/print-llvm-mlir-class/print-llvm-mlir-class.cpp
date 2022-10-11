#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>
#include <string>

template <typename T> inline std::string llvm_sprint(T &&val) {
  std::string buf;
  llvm::raw_string_ostream rso(buf);
  val.print(rso);
  return std::move(buf);
}

void print_types() {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  {
    // vector<4x8xf32>
    mlir::Type type = mlir::VectorType::get({4, 8}, builder.getF32Type());
    std::cout << llvm_sprint(type) << std::endl;
    std::cout << type.isa<mlir::VectorType>() << std::endl;
  }

  return;
}

void print_affine_map() {
  mlir::MLIRContext context;
  mlir::OpBuilder builder(&context);

  {
    auto affineMap = builder.getMultiDimIdentityMap(4);
    std::cout << llvm_sprint(affineMap) << std::endl;
  }

  {
    llvm::SmallVector<mlir::AffineExpr, 2> affineExprs{
        mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Add,
                                    builder.getAffineDimExpr(1),
                                    mlir::getAffineSymbolExpr(1, &context)),
        builder.getAffineConstantExpr(0)};
    auto affineMap = mlir::AffineMap::get(3, 3, affineExprs, &context);
    std::cout << llvm_sprint(affineMap) << std::endl;
  }

  return;
}

void print_mlir_ops() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::arith::ArithmeticDialect>();
  mlir::OpBuilder builder(&context);

  {
    auto value = builder.create<mlir::arith::ConstantIndexOp>(
        builder.getUnknownLoc(), 0);
    std::cout << llvm_sprint(value) << std::endl;
  }

  {
    std::vector<float> values{1.f, 2.f, 3.f, 3.f};
    auto dataType = mlir::RankedTensorType::get({2, 2}, builder.getF32Type());
    auto dataAttribute = mlir::DenseElementsAttr::get(
        dataType, llvm::ArrayRef<float>{values.data(), values.size()});
    auto value = builder.create<mlir::arith::ConstantOp>(
        builder.getUnknownLoc(), dataAttribute);
    std::cout << llvm_sprint(value) << std::endl;
  }

  return;
}

int main() {
  print_types();
  return 0;
}
