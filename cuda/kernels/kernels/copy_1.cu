#include <cstdint>

#include "c10/cuda/CUDAStream.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "torch/all.h"

#include "../module.h"
#include "../support/assert.h"

namespace {

template <int FEATURE_SIZE, int NUM_THREADS>
__global__ void copy_1(  //
    half* dst,           // [n, c, h, w]
    const half* src,     // [n, c, h, w]
    int stride_n,        //
    int stride_c,        //
    int stride_h,        //
    int h,               //
    int w                //
) {
  assert(FEATURE_SIZE == w && "feature_size != w");
  int idx_n = blockIdx.x;
  int idx_c = blockIdx.y;

  constexpr int WARP_SIZE = 32;
  static_assert(NUM_THREADS % WARP_SIZE == 0);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  for (int idx_h = warp; idx_h < h; idx_h += NUM_WARPS) {
    auto dst_ptr = dst + idx_n * stride_n + idx_c * stride_c + idx_h * stride_h;
    auto src_ptr = src + idx_n * stride_n + idx_c * stride_c + idx_h * stride_h;

    using LOAD_T = uint16_t;
    constexpr int VEC_SIZE = sizeof(LOAD_T) / sizeof(half);
    constexpr int NUM_LOADS = FEATURE_SIZE / (WARP_SIZE * VEC_SIZE);
#pragma unroll
    for (int i = 0; i < NUM_LOADS; ++i) {
      const int offset = (i * WARP_SIZE + lane) * VEC_SIZE;
      *reinterpret_cast<LOAD_T*>(dst_ptr + offset) = *reinterpret_cast<const LOAD_T*>(src_ptr + offset);
    }
  }
}

void copy_1_launcher(torch::Tensor& dst, const torch::Tensor& src) {
  cr::cr_assert(dst.is_contiguous() && src.is_contiguous(), "not contiguous");
  cr::cr_assert(src.size(2) == 1024, "");
  cr::cr_assert(src.size(3) == 1024, "");

  int n = src.size(0);
  int c = src.size(1);
  int stride_n = src.stride(0);
  int stride_c = src.stride(1);
  int stride_h = src.stride(2);
  dim3 grids(n, c);

  half* dst_ptr = reinterpret_cast<half*>(dst.data_ptr());
  const half* src_ptr = reinterpret_cast<const half*>(src.data_ptr());
  copy_1<1024, 128><<<grids, 128>>>(dst_ptr, src_ptr, stride_n, stride_c, stride_h, 1024, 1024);
}

static cr::Register _([](pybind11::module& m) { m.def("copy_1", &copy_1_launcher); });

}  // namespace
