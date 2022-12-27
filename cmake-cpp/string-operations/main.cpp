#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<std::string> split_string(const std::string &src, char delimiter) {
  std::stringstream ss(src);
  std::vector<std::string> ret;
  while (ss.good()) {
    std::string substr;
    std::getline(ss, substr, delimiter);
    ret.push_back(substr);
  }
  return ret;
}

int main(int argc, char *argv[]) {
  std::string shapes = argv[1];

  for (const auto &s : split_string(shapes, ','))
    std::cout << s << std::endl;

  return 0;
}