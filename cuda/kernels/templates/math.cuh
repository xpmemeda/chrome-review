#pragma once

#include <cstdint>

#include "cuda_fp16.h"

namespace cr {

namespace f16_detail {

inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

template <typename T>
inline __device__ float sum(T v);

template <>
inline __device__ float sum<uint16_t>(uint16_t v) {
  float f;
  asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(v));
  return f;
}

template <>
inline __device__ float sum<uint32_t>(uint32_t v) {
  uint16_t lo, hi;
  asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
  return sum(lo) + sum(hi);
}

template <>
inline __device__ float sum<uint2>(uint2 v) {
  uint32_t c = add(v.x, v.y);
  return sum(v);
}

template <>
inline __device__ float sum<uint4>(uint4 v) {
  uint32_t c = add(v.x, v.y);
  c = add(c, v.z);
  c = add(c, v.w);
  return sum(c);
}

}  // namespace f16_detail

using half1 = __half;
using half2 = __half2;
struct half4 {
  half2 x;
  half2 y;
};
struct half8 {
  half2 x;
  half2 y;
  half2 z;
  half2 w;
};

inline __device__ half1 add(half1 a, half1 b) { return a + b; }
inline __device__ half2 add(half2 a, half2 b) { return make_half2(a.x + b.x, a.y + b.y); }
inline __device__ half4 add(half4 a, half4 b) {
  half4 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  return c;
}
inline __device__ half8 add(half8 a, half8 b) {
  half8 c;
  c.x = add(a.x, b.x);
  c.y = add(a.y, b.y);
  c.z = add(a.z, b.z);
  c.w = add(a.w, b.w);
  return c;
}

using Float1 = float;
using Float2 = float2;
struct Float4 {
  float2 x;
  float2 y;
};
struct Float8 {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

template <typename T>
struct ReduceTrait {};

template <>
struct ReduceTrait<half1> {
  using type = Float1;
};
template <>
struct ReduceTrait<half2> {
  using type = Float2;
};
template <>
struct ReduceTrait<half4> {
  using type = Float4;
};
template <>
struct ReduceTrait<half8> {
  using type = Float8;
};

template <typename T, typename U = typename ReduceTrait<T>::type>
inline __device__ U sum(T v);

template <>
inline __device__ Float1 sum(half1 v) {
  // float f;
  // asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(v));
  // return f;
  return __half2float(v);
}

struct F16Operator {
  static inline __device__ uint32_t add(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
  }

  /**
   * @brief Sums all the fp16 elements.
   *
   * @tparam Vec half1, half2, half3, half4.
   */
  template <typename Vec>
  static inline __device__ float sum(Vec v) {
    return f16_detail::sum(v);
  }
};

}  // namespace cr
