#include <boost/stacktrace.hpp>
#include <iostream>

void fn() {
  std::string stacktrace = boost::stacktrace::to_string(boost::stacktrace::stacktrace());
  std::cout << stacktrace << std::endl;
}

int main() {
  fn();
  return 0;
}