#include <fstream>
#include <string>
#include <vector>

using token_t = int64_t;

void append_int_list(const std::vector<token_t>& ints, const std::string& path) {
  std::ofstream out(path, std::ios::app | std::ios::binary);

  std::string line;
  line.reserve(ints.size() * 12);

  for (size_t i = 0; i < ints.size(); i++) {
    line.append(std::to_string(ints[i]));
    if (i + 1 < ints.size()) {
      line.push_back(' ');
    }
  }
  line.push_back('\n');

  out.write(line.data(), line.size());
}

int main() {
  append_int_list({9, 7, 23843, 2, 88}, "/tmp/xxx.txt");
  append_int_list({9, 7, 23843, 2, 88, 9}, "/tmp/xxx.txt");

  return 0;
}