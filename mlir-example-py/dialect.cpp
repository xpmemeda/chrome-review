#include "dialect.h"

#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::tfcc;

#include "dialect.cpp.inc"

void TfccDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "ops.cpp.inc"
    >();
}

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       double value) {
  auto dataType = builder.getF64Type();
  auto dataAttribute = FloatAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}

void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getF64Type());
  state.addOperands({lhs, rhs});
}

void MulOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                  mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(builder.getF64Type());
  state.addOperands({lhs, rhs});
}

#define GET_OP_CLASSES
#include "ops.cpp.inc"
