#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define E2_2 E2_0 E2_0 E2_0 E2_0
#define E2_4 E2_2 E2_2 E2_2 E2_2
#define E2_6 E2_4 E2_4 E2_4 E2_4
#define E2_8 E2_6 E2_6 E2_6 E2_6
#define E2_10 E2_8 E2_8 E2_8 E2_8
#define E2_12 E2_10 E2_10 E2_10 E2_10
#define E2_14 E2_12 E2_12 E2_12 E2_12
#define E2_16 E2_14 E2_14 E2_14 E2_14
#define E2_18 E2_16 E2_16 E2_16 E2_16
#define E2_20 E2_18 E2_18 E2_18 E2_18

void fun() {
  // 8000times.
  {
    int c = 100, a = 1;
    auto t1 = std::chrono::system_clock::now();
    for (int i = 0; i < 2 << 14; ++i)
      ;
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
  }
  {
    int c = 100, a = 1;
#define E2_0 c = c + a;
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 50 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
    int c = 100, a = 1;
    auto t1 = std::chrono::system_clock::now();
    for (int i = 0; i < 2 << 14; ++i)
      c = c + a;
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
  }
  {
    int c = 100, a = 1;
#define E2_0 c = c * a;
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 79 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
    float c = 100.0, a = 1.1;
#define E2_0 c = c + a;
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 99 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
    float c = 100.0, a = 1.1;
#define E2_0 c = c * a;
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 102 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
    float c = 100.0, a = 1.1;
#define E2_0 c = c / a;
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 870 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
#define E2_0 std::chrono::system_clock::now();
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 1150 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
    std::vector<int> vec;
#define E2_0 vec.push_back(0);
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 1150 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
  {
    std::vector<int> vec;
    vec.reserve(2 << 14);
#define E2_0 vec.push_back(0);
    auto t1 = std::chrono::system_clock::now();
    E2_14; // 1150 micros
    auto t2 = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
#undef E2_0
  }
}

int main() {
  fun();
  return 0;
}