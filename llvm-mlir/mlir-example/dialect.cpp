#include "dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::demo;

#include "dialect.cpp.inc"

void DemoDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "ops.cpp.inc"
    >();
}

#define GET_OP_CLASSES
#include "ops.cpp.inc"
