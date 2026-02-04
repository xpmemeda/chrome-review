#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <cutlass/platform/platform.h>

namespace {

constexpr int WARP_SIZE = 32;

/**
 * topk[0] > topk[1] > ... > topk[K-1]
 */
template <typename scalar_t, int K>
__device__ __forceinline__ void merge_topk(cutlass::Array<scalar_t, K>& topk, scalar_t value) {
#pragma unroll
  for (int j = 0; j < K; ++j) {
    if (value > topk[j]) {
      auto tmp = topk[j];
      topk[j] = value;
      value = tmp;
    }
  }
}

/**
 * lane 0 has the topk results after this function.
 * this function requires K * WARP_SIZE * sizeof(scalar_t) / 2 bytes of shared memory.
 */
template <typename scalar_t, int K>
__device__ __forceinline__ void warp_topk(cutlass::Array<scalar_t, K>& topk, scalar_t* warp_smem) {
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane_idx = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int i = WARP_SIZE; i > 1; i /= 2) {
    int mid = i / 2;

    // upper lanes write values to shared memory.
    if (lane_idx >= mid && lane_idx < i) {
      scalar_t* lane_smem_ptr = warp_smem + (lane_idx - mid) * K;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        lane_smem_ptr[j] = topk[j];
      }
    }

    __syncwarp();

    // lower lanes read values from shared memory and merge.
    if (lane_idx < mid) {
      scalar_t* lane_smem_ptr = warp_smem + lane_idx * K;
#pragma unroll
      for (int j = 0; j < K; ++j) {
        merge_topk(topk, lane_smem_ptr[j]);
      }
    }

    __syncwarp();
  }
}

/**
 * this function requires blockDim.x * K * sizeof(scalar_t) / 2 bytes of shared memory.
 */
template <typename scalar_t, int K>
__device__ __forceinline__ void cta_topk(cutlass::Array<scalar_t, K>& topk, scalar_t* smem) {
  constexpr int NUM_ELEMENTS_PER_WARP = K * WARP_SIZE;
  constexpr int REQUIRED_SMEM_NUMEL_PER_WARP = NUM_ELEMENTS_PER_WARP / 2;
  const int num_warps = blockDim.x / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane_idx = threadIdx.x % WARP_SIZE;

  warp_topk(topk, smem + warp_idx * NUM_ELEMENTS_PER_WARP);

  for (int i = num_warps; i > 1; i /= 2) {
    int mid = i / 2;
    // upper warps write values to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      scalar_t* warp_smem_ptr = smem + (warp_idx - mid) * K;
      if (lane_idx == 0) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
          warp_smem_ptr[j] = topk[j];
        }
      }
    };

    __syncthreads();

    // lower warps read values from shared memory and merge.
    if (warp_idx < mid) {
      scalar_t* warp_smem_ptr = smem + warp_idx * K;
      if (lane_idx == 0) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
          merge_topk(topk, warp_smem_ptr[j]);
        }
      }
    };

    __syncthreads();
  }
}

template <typename scalar_t, int K>
__global__ void topk_kernel(scalar_t* src, scalar_t* dst, int M, int N) {
  src = src + blockIdx.x * N;
  dst = dst + blockIdx.x * K;

  cutlass::Array<scalar_t, K> topk;
  scalar_t neg_inf = -cutlass::platform::numeric_limits<scalar_t>::infinity();
#pragma unroll
  for (int j = 0; j < K; ++j) {
    topk[j] = neg_inf;
  }

  for (unsigned int i = threadIdx.x; i < N; i += blockDim.x) {
    scalar_t val = src[i];
    for (int j = 0; j < K; ++j) {
      if (val > topk[j]) {
        auto tmp = topk[j];
        topk[j] = val;
        val = tmp;
      }
    }
  }

  extern __shared__ scalar_t smem[];
  cta_topk(topk, smem);

  cutlass::NumericConverter<float, scalar_t> cvt;
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < K; ++j) {
      printf("%f\n", cvt(topk[j]));
      dst[j] = topk[j];
    }
  }
}

void topk(at::Tensor& src, at::Tensor& dst, int k) {
  const int M = src.size(0);
  const int N = src.size(1);

  const int threads = 256;
  const int blocks = M;
  const int shared_mem_bytes = threads * k / 2 * src.element_size();

  auto stream = at::cuda::getCurrentCUDAStream();

  using scalar_t = cutlass::half_t;

  if (k == 32) {
    topk_kernel<scalar_t, 32><<<blocks, threads, shared_mem_bytes, stream.stream()>>>(
        reinterpret_cast<scalar_t*>(src.data_ptr()), reinterpret_cast<scalar_t*>(dst.data_ptr()), M, N);
    auto r = cudaDeviceSynchronize();
    if (r != cudaSuccess) {
      throw std::runtime_error("cuda err " + std::to_string(r));
    }
  } else {
    throw std::runtime_error("k != 32.");
  }
}

}  // namespace

#include "module.h"
static cr::Register _([](pybind11::module& m) { m.def("topk", &topk); });
