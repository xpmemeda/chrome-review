#include "./utils.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

std::string exec(const char *cmd) {
  std::array<char, 128> buffer;
  std::string result;
  std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd, "r"), pclose);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

int64_t get_time_stamp() {
  auto now = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             now.time_since_epoch())
      .count();
}

std::string load_data_from_file(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("open file: \"" + path + "\" failed");
  ifs.seekg(0, ifs.end);
  auto length = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> buffer(new char[length]);
  ifs.read(buffer.get(), length);
  std::string data(buffer.get(), length);
  return data;
}

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