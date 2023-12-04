#include <math.h>
#include <chrono>
#include <functional>
#include <iostream>
#include <vector>

#include "softmax.h"

std::vector<float> softmax_naive(const std::vector<float>& src) {
  float exp_sum;
  std::vector<float> r(src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    float v = expf(src[i]);
    exp_sum += v;
    r[i] = v;
  }
  for (size_t i = 0; i < src.size(); ++i) {
    r[i] = r[i] / exp_sum;
  }
  return r;
}

int main() {
  std::vector<float> src(1024);
  std::generate(src.begin(), src.end(), [&]() { return static_cast<float>(rand()) / RAND_MAX; });

  std::vector<float> naive_r = softmax_naive(src);

  {
    std::cout << "Testing softmax v1...." << std::endl;
    std::vector<float> cuda_v1_r = softmax_v1(src);
    std::cout << "[correctness]" << (std::equal(naive_r.begin(), naive_r.end(), cuda_v1_r.begin()) ? "True" : "False")
              << std::endl;
  }

  {
    std::cout << "Testing softmax v2...." << std::endl;
    std::vector<float> cuda_v2_r = softmax_v2(src);
    std::cout << "[correctness]" << (std::equal(naive_r.begin(), naive_r.end(), cuda_v2_r.begin()) ? "True" : "False")
              << std::endl;
  }

  {
    std::cout << "Testing softmax v3...." << std::endl;
    std::vector<float> cuda_v3_r = softmax_v3(src);
    std::cout << "[correctness]" << (std::equal(naive_r.begin(), naive_r.end(), cuda_v3_r.begin()) ? "True" : "False")
              << std::endl;
  }

  return 0;
}