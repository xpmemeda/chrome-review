#include <cstdint>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

namespace cr {

namespace {

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

// FP32 accumulator vector types corresponding to Vec.
struct Float4_ {
  float2 x;
  float2 y;
};

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};
template <typename T>
struct FloatVec {};
template <>
struct FloatVec<uint16_t> {
  using Type = float;
};
template <>
struct FloatVec<uint32_t> {
  using Type = float2;
};
template <>
struct FloatVec<uint2> {
  using Type = Float4_;
};
template <>
struct FloatVec<uint4> {
  using Type = Float8_;
};

struct F16Operator {
  inline __device__ uint32_t add(uint32_t a, uint32_t b) {
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
  }

  /**
   * @brief Sums all the fp16 elements.
   *
   * @tparam Vec The type used to represent fp16 elements vector.
   *           Such as uint16_t, uint32_t, uint2 and uint4.
   */
  template <typename Vec>
  inline __device__ float sum(Vec v) {
    return f16_detail::sum(v);
  }
};

}  // namespace

}  // namespace cr
