#include <fstream>
#include <iostream>
#include <string>

std::string generate_cpp_header() { return R"(int Add(int x, int y);)"; }

std::string generate_cpp_body() {
  return R"(int Add(int x, int y) {
    return x + y;
})";
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "usage: " << argv[0] << " <cpp header file path> <cpp body file path>";
    return 1;
  }
  std::string cpp_header_file_path = argv[1];
  std::string cpp_body_file_path = argv[2];
  std::ofstream header_ofs(cpp_header_file_path);
  header_ofs << generate_cpp_header();
  std::ofstream body_ofs(cpp_body_file_path);
  body_ofs << generate_cpp_body();
  return 0;
}