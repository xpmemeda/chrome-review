#include <experimental/filesystem>
#include <iostream>

using namespace std::experimental;

int main() {
  const char *path = "../main.cpp";

  if (filesystem::exists(path)) {
    std::cout << filesystem::is_directory(path) << std::endl;
    std::cout << filesystem::is_regular_file(path) << std::endl;
    std::cout << filesystem::file_size(path) << std::endl;
  } else
    std::cout << "not found main.cpp" << std::endl;
  std::cout << filesystem::current_path() << std::endl;

  for (const auto &dirEntry : filesystem::recursive_directory_iterator("."))
    std::cout << dirEntry.path().c_str() << std::endl;

  for (const auto &dirEntry : filesystem::directory_iterator(".."))
    std::cout << dirEntry << std::endl;
  return 0;
}
