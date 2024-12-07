#include <fstream>
#include <iostream>

#include "cr.pb.h"

int main() {
  cr::Cr x;
  x.set_string(R"("你好")");
  x.set_bytes(R"("你好")");
  x.set_int64(13);

  std::ofstream cpp("cpp.pb");
  cpp << x.SerializeAsString();
  return 0;
}