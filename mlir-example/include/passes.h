#pragma once

#include <memory>

namespace mlir {
class Pass;

namespace demo {

std::unique_ptr<Pass> createLowerToVectorPass();
std::unique_ptr<Pass> createLowerToLinalgPass();
std::unique_ptr<Pass> createLowerToLlvmPass();

}

}
