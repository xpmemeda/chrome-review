#pragma once

#include <cstdio>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace cr {

struct HalfPrinter {
  static __device__ inline void print(half val) {
    auto v = __half2float(val);
    printf("%f\n", v);
  }

  static __device__ inline void print(uint4 val) {
    auto h0 = __half2float(__ushort_as_half(val.x & 0xFFFF));
    auto h1 = __half2float(__ushort_as_half((val.x >> 16) & 0xFFFF));
    auto h2 = __half2float(__ushort_as_half(val.y & 0xFFFF));
    auto h3 = __half2float(__ushort_as_half((val.y >> 16) & 0xFFFF));
    auto h4 = __half2float(__ushort_as_half(val.z & 0xFFFF));
    auto h5 = __half2float(__ushort_as_half((val.z >> 16) & 0xFFFF));
    auto h6 = __half2float(__ushort_as_half(val.w & 0xFFFF));
    auto h7 = __half2float(__ushort_as_half((val.w >> 16) & 0xFFFF));
    printf("%f,%f,%f,%f,%f,%f,%f,%f\n", h0, h1, h2, h3, h4, h5, h6, h7);
  }
};

}  // namespace cr
