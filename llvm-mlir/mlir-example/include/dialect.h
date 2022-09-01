#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "dialect.h.inc"

#define GET_OP_CLASSES
#include "ops.h.inc"
