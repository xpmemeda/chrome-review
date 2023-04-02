#include "gperftools/profiler.h"
#include <string>

int foo() {
  std::string x(1 << 30, '\0');
  std::string y(1 << 30, '\0');
  y.reserve(1u << 31);
  return 0;
}

int main() { return foo(); }
