#include <set>
#include <string>
#include <vector>

namespace cr::stl {

std::vector<std::string> split(const std::string& src, char delimiter);

std::string load_data_from_file(const std::string& path);
void save_data_to_file(const std::string& path, const std::string& data);

template <typename T>
std::string convert_vec_to_string(const std::vector<T>& vec) {
  if (vec.empty()) {
    return "()";
  }
  std::string ret = "(";
  for (auto v : vec) {
    ret.append(std::to_string(v) + ", ");
  }
  ret.pop_back();
  ret.pop_back();
  ret.append(")");
  return ret;
}

}  // namespace cr::stl
