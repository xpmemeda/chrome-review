#include <iostream>

std::string bytes_to_hex(const std::string &bytes) {
  static const char hex_digits[] = "0123456789ABCDEF";
  std::string hex;
  hex.reserve(bytes.size() * 2);
  for (unsigned char c : bytes) {
    hex.push_back(hex_digits[c >> 4]);
    hex.push_back(hex_digits[c & 15]);
  }
  return hex;
}

std::string hex_to_bytes(const std::string &hex) {
  std::string bytes;
  for (size_t i = 0; i < hex.size(); i += 2) {
    bytes.push_back(
        static_cast<unsigned char>(strtol(hex.substr(i, 2).c_str(), NULL, 16)));
  }
  return bytes;
}

int main() {
  auto h = bytes_to_hex("hello world");
  std::cout << h << std::endl;
  auto b = hex_to_bytes(h);
  std::cout << b << std::endl;
  return 0;
}
