#include "string_utils.h"
#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace comm {

int string_printf_impl(std::string& output, const char* format, va_list args) {
  const int write_point = output.size();
  int remaining = output.capacity() - write_point;
  output.resize(output.capacity());

  va_list copied_args;
  va_copy(copied_args, args);
  int bytes_used = vsnprintf(&output[write_point], remaining, format, copied_args);
  va_end(copied_args);
  if (bytes_used < 0) {
    return -1;
  } else if (bytes_used < remaining) {
    output.resize(write_point + bytes_used);
  } else {
    output.resize(write_point + bytes_used + 1);
    remaining = bytes_used + 1;
    bytes_used = vsnprintf(&output[write_point], remaining, format, args);
    if (bytes_used + 1 != remaining) {
      return -1;
    }
    output.resize(write_point + bytes_used);
  }
  return 0;
}

std::string StringPrintf(const char* format, ...) {
  std::string ret;
  ret.reserve(std::max(32UL, strlen(format) * 10));

  va_list ap;
  va_start(ap, format);
  if (string_printf_impl(ret, format, ap) != 0) {
    ret.clear();
  }
  va_end(ap);
  return ret;
}

std::string BinaryToHex(const std::string& binary_data) {
  std::ostringstream oss;
  oss << std::hex << std::uppercase << std::setfill('0');
  for (unsigned char byte : binary_data) {
    oss << std::setw(2) << static_cast<int>(byte);
  }
  return oss.str();
}

std::string HexToBinary(const std::string& hex_str) {
  assert(hex_str.size() % 2 == 0);
  std::string result;
  result.reserve(hex_str.size() / 2);
  for (size_t i = 0; i < hex_str.length(); i += 2) {
    std::string byte_str = hex_str.substr(i, 2);
    result.push_back(static_cast<char>(std::stoul(byte_str, nullptr, 16)));
  }
  return result;
}

uint64_t SimpleHashString(const std::string& str) {
  const uint64_t fnv_prime = 1099511628211ULL;
  const uint64_t fnv_offset_basis = 14695981039346656037ULL;

  uint64_t hash = fnv_offset_basis;

  for (char c : str) {
    hash ^= static_cast<uint64_t>(c);
    hash *= fnv_prime;
  }

  return hash;
}

}  // namespace comm