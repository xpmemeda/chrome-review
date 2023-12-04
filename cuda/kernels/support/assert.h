#pragma once

#include <stdio.h>
#include <exception>
#include <memory>
#include <string>

namespace cr {

template <typename... Args>
std::string cr_formats(const std::string& format, Args&&... args) {
  int size = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

template <typename... Args>
void cr_assert(bool val, const std::string& err_fmt, Args&&... args) {
  if (!val) {
    throw std::runtime_error(cr_formats(err_fmt, std::forward<Args>(args)...));
  }
}

}  // namespace cr