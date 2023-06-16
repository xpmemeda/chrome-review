#include <ctime>
#include <iostream>
#include <string>

std::string format_time(time_t now) {
  char buf[32];
  strftime(buf, 32, "%c", localtime(&now));
  return std::string(buf);
}

time_t parse_format_time(const char* format_time) {
  tm t;
  strptime(format_time, "%c", &t);
  return std::mktime(&t);
}

int main() {
  time_t now = time(nullptr);
  std::string fmt_time = format_time(now);
  std::cout << fmt_time << std::endl;
  std::cout << (now == parse_format_time(fmt_time.c_str())) << std::endl;
  return 0;
}