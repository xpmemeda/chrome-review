#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace tfcc {
std::unique_ptr<Pass> createLowerToLlvmPass();
}

}
