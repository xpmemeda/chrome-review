#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "rnpz.h"

int main() {
  rnpz::NpyArray array(std::vector<size_t>({2}), rnpz::NpyArray::ElementTypeID::INT64, std::make_unique<char[]>(16));
  rnpz::npz_t npz_file;
  npz_file.insert({std::string("x"), std::move(array)});
  rnpz::save_npz(npz_file, "x.npz");

  auto r = rnpz::load_npz("x.npz");
  for (auto it = r.begin(); it != r.end(); ++it) {
    std::cout << it->first << std::endl;
    std::cout << (int64_t)it->second.getTypeID() << std::endl;
    std::cout << *(int64_t*)(it->second.getData()) << std::endl;
  }

  return 0;
}