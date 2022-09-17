#pragma once

#include "memory"
#include <cstdint>
#include <sstream>
#include <string>

std::string exec(const char *cmd);
int64_t get_time_stamp();
std::string load_data_from_file(const std::string &path);
std::string bytes_to_hex(const std::string &bytes);
std::string hex_to_bytes(const std::string &hex);

template <typename... Args>
static std::string string_format(const std::string &format, Args... args) {
  int size_s = std::snprintf(nullptr, 0, format.c_str(), args...) + 1;
  if (size_s <= 0) {
    throw std::runtime_error("Error during formatting.");
  }
  auto size = static_cast<size_t>(size_s);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args...);
  return std::string(buf.get(), buf.get() + size - 1);
}

class my_exception : public std::runtime_error {
  std::string msg;

public:
  my_exception(const std::string &arg, const char *file, int line)
      : std::runtime_error(arg) {
    std::ostringstream o;
    o << file << ":" << line << ": " << arg;
    msg = o.str();
  }
  ~my_exception() throw() {}
  const char *what() const throw() { return msg.c_str(); }
};
#define my_throw(arg) throw my_exception(arg, __FILE__, __LINE__);
#define my_assert(condition, arg)                                              \
  if (!(condition))                                                            \
    my_throw(arg);
