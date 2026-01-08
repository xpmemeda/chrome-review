#include "rediscpp.h"

#include <chrono>
#include <iostream>

template <typename C>
void print_vec(const C& c) {
  for (auto& v : c) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}

int main() {
  CppRedis cpp_redis("127.0.0.1", 6379, std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
  cpp_redis.set("x", "y", 10, false);
  cpp_redis.set("y", "z", 10, false);

  std::cout << cpp_redis.get("x") << std::endl;

  print_vec(cpp_redis.mget({"x", "y"}));

  cpp_redis.msadd(std::vector<std::string>{"tag0", "tag1", "tag2"}, "addr0");
  cpp_redis.msadd(std::vector<std::string>{"tag0", "tag1"}, "addr1");

  auto members = cpp_redis.msget({"tag0", "tag1", "tag2", "tag3"});
  for (auto& member : members) {
    print_vec(member);
  }

  cpp_redis.msrem({"tag0", "tag1"}, "addr1");
  members = cpp_redis.msget({"tag0", "tag1", "tag2", "tag3"});
  for (auto& member : members) {
    print_vec(member);
  }

  return 0;
}