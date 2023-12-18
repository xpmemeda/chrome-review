#include "stl.h"

#include <fstream>
#include <memory>
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

std::string load_data_from_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("open file: \"" + path + "\" failed");
  }
  ifs.seekg(0, ifs.end);
  auto length = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> buffer(new char[length]);
  ifs.read(buffer.get(), length);
  std::string data(buffer.get(), length);
  return data;
}

void save_data_to_file(const std::string& path, const std::string& data) {
  std::ofstream ofs(path, std::ios::binary);
  if (!ofs) {
    throw std::runtime_error("open file: \"" + path + "\" failed");
  }
  ofs << data;
  ofs.close();
}

}  // namespace cr::stl
