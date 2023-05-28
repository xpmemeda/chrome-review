#include "immintrin.h"
#include <iostream>

extern "C" {
extern __m256 __svml_expf8(__m256);
}

int main() {
  auto x = _mm256_set1_ps(1.f);
  auto y = __svml_expf8(x);
  return 0;
}