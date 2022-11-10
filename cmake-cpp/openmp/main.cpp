#include "omp.h"
#include <iostream>

int check_openmp() {
#if _OPENMP
  std::cout << "support openmp " << std::endl;
#else
  std::cout << "not support openmp" << std::endl;
#endif
  return 0;
}

int print() {
#pragma omp parallel for num_threads(4)
  for (int i = 0; i < 10; i++) {
    std::cout << i << std::endl;
  }
  return 0;
}

int sum() {
  int sum = 0;
#pragma omp parallel for num_threads(32) reduction(+ : sum)
  for (int i = 0; i < 100; i++) {
    sum += i;
  }

  std::cout << sum << std::endl;
}

// https://zhuanlan.zhihu.com/p/61857547
int main() {
  check_openmp();
  print();
  sum();
  return 0;
}
