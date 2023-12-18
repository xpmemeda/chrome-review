#include "stl.h"

#include <sstream>

namespace cr::stl {

std::vector<std::string> split(const std::string& src, char delimiter) {
  std::stringstream ss(src);
  std::vector<std::string> ret;
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, delimiter);
    ret.push_back(substr);
  }
  return ret;
}

}  // namespace cr::stl
