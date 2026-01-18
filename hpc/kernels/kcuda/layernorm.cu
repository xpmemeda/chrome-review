#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "ATen/cuda/CUDAContext.h"
#include "torch/all.h"
#include "torch/csrc/autograd/python_variable.h"

#include "./module.h"
#include "./support/assert.h"

namespace {

struct Add {
  static __device__ __forceinline__ float init() { return 0.f; }
  static __device__ __forceinline__ float apply(float a, float b) { return a + b; }
};

template <typename Op>
inline __device__ float reduce(float* smem, float v, bool print) {
  constexpr unsigned WARP_SIZE = 32;
  for (unsigned s = WARP_SIZE / 2; s >= 1; s /= 2) {
    v = Op::apply(v, __shfl_xor_sync(0xffffffff, v, s, WARP_SIZE));
  }
  // if (print) {
  //   printf("thread_id %u, v1: %f\n", threadIdx.x, v);
  // }
  unsigned warp_id = threadIdx.x / WARP_SIZE;
  unsigned lane_id = threadIdx.x % WARP_SIZE;
  if (lane_id == 0) {
    smem[warp_id] = v;
  }
  __syncthreads();
  unsigned num_warps = blockDim.x / WARP_SIZE;
  v = Op::init();
  for (unsigned i = lane_id; i < num_warps; i += WARP_SIZE) {
    v = Op::apply(v, smem[i]);
  }

  // if (print) {
  //   printf("thread_id %u, v2: %f\n", threadIdx.x, v);
  // }

  __syncthreads();
  for (unsigned s = WARP_SIZE / 2; s >= 1; s /= 2) {
    v = Op::apply(v, __shfl_xor_sync(0xffffffff, v, s, WARP_SIZE));
  }

  // if (print) {
  //   printf("thread_id %u, v3: %f\n", threadIdx.x, v);
  // }
  return v;
}

template <typename>
constexpr bool always_false_v = false;

template <typename T>
__device__ inline float convert_to_float(T v) {
  float xi = 0.f;
  if constexpr (std::is_same_v<T, __half>) {
    xi = __half2float(v);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    xi = __bfloat162float(v);
  } else if constexpr (std::is_same_v<T, float>) {
    xi = v;
  } else {
    static_assert(always_false_v<T>);
  }
  return xi;
}

template <typename T>
__device__ inline T convert_from_float(float v) {
  T x;
  if constexpr (std::is_same_v<T, __half>) {
    x = __float2half(v);
  } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    x = __float2bfloat16(v);
  } else if constexpr (std::is_same_v<T, float>) {
    x = v;
  } else {
    static_assert(always_false_v<T>);
  }
  return x;
}

template <typename T>
__global__ void layernorm_kernel_v1(T* x, T* out, T* w, T* b, unsigned m, unsigned n, float eps) {
  // __shared__ float* smem;
  extern __shared__ float smem[];

  x = x + blockIdx.x * n;
  out = out + blockIdx.x * n;

  float mean = 0;
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    mean += convert_to_float<T>(x[i]);
  }
  mean = reduce<Add>(smem, mean, true);
  mean = mean / n;

  // printf("mean: %f\n", mean);

  float var = 0;
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    float xi = convert_to_float<T>(x[i]);
    float xi_sub_mean = xi - mean;
    var += xi_sub_mean * xi_sub_mean;
  }
  var = reduce<Add>(smem, var, false);
  var = var / n;

  // printf("var: %f\n", var);
  float rstd = 1 / sqrt(var + eps);
  // printf("rstd: %f\n", rstd);

  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    float v = (convert_to_float<T>(x[i]) - mean) * rstd;
    v = v * convert_to_float(w[i]) + convert_to_float(b[i]);
    out[i] = convert_from_float<T>(v);
  }

  return;
}

struct WelfordMeanVarComputer {
  static constexpr unsigned WARP_SIZE = 32;

  float mean_;
  float m2_;
  unsigned n_;

  __forceinline__ __device__ WelfordMeanVarComputer() : mean_(0.f), m2_(0.f), n_(0) {}

  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  __forceinline__ __device__ void update(float x) {
    n_ += 1;
    float mean = mean_ + (x - mean_) / n_;
    m2_ = m2_ + (x - mean_) * (x - mean);
    mean_ = mean;
  }

  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
  __forceinline__ __device__ void warpReduce() {
    for (unsigned s = WARP_SIZE / 2; s >= 1; s /= 2) {
      float b_mean = __shfl_xor_sync(0xffffffff, mean_, s, WARP_SIZE);
      float b_m2 = __shfl_xor_sync(0xffffffff, m2_, s, WARP_SIZE);
      unsigned b_n = __shfl_xor_sync(0xffffffff, n_, s, WARP_SIZE);

      if (b_n == 0) {
        continue;
      }

      unsigned n = n_ + b_n;
      float delta = b_mean - mean_;
      float mean = mean_ + delta * b_n / n;
      float m2 = m2_ + b_m2 + delta * delta * n_ * b_n / n;

      mean_ = mean;
      m2_ = m2;
      n_ = n;
    }
  }

  __forceinline__ __device__ void blockReduce() {
    warpReduce();

    __shared__ float smem_mean[32];
    __shared__ float smem_m2[32];
    __shared__ unsigned smem_n[32];
    __shared__ float mean_broadcast;
    __shared__ float m2_broadcast;
    __shared__ unsigned n_broadcast;

    unsigned num_warps = blockDim.x / WARP_SIZE;
    unsigned warp_id = threadIdx.x / WARP_SIZE;
    unsigned lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
      smem_mean[warp_id] = mean_;
      smem_m2[warp_id] = m2_;
      smem_n[warp_id] = n_;
    }
    __syncthreads();

    if (warp_id == 0) {
      if (lane_id < num_warps) {
        mean_ = smem_mean[lane_id];
        m2_ = smem_m2[lane_id];
        n_ = smem_n[lane_id];
      } else {
        mean_ = 0.f;
        m2_ = 0.f;
        n_ = 0;
      }
      __syncwarp();

      warpReduce();

      if (lane_id == 0) {
        mean_broadcast = mean_;
        m2_broadcast = m2_;
        n_broadcast = n_;
      }
    }
    __syncthreads();

    mean_ = mean_broadcast;
    m2_ = m2_broadcast;
    n_ = n_broadcast;
  }

  __forceinline__ __device__ float getMean() { return mean_; }
  __forceinline__ __device__ float getVar() { return m2_ / n_; }
};

template <typename T>
__global__ void layernorm_kernel_v2(T* x, T* out, T* w, T* b, unsigned m, unsigned n, float eps) {
  // __shared__ float* smem;
  extern __shared__ float smem[];

  x = x + blockIdx.x * n;
  out = out + blockIdx.x * n;

  WelfordMeanVarComputer mean_var_computer;
  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    mean_var_computer.update(convert_to_float<T>(x[i]));
  }
  mean_var_computer.blockReduce();

  float mean = mean_var_computer.getMean();
  float var = mean_var_computer.getVar();
  float rstd = 1 / sqrt(var + eps);

  for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
    float v = (convert_to_float<T>(x[i]) - mean) * rstd;
    v = v * convert_to_float(w[i]) + convert_to_float(b[i]);
    out[i] = convert_from_float<T>(v);
  }
}

template <class T, unsigned NUM_ROWS_PER_WARP, unsigned N>
__global__ void layernorm_kernel_v3(T* x, T* out, T* w, T* b, unsigned m, unsigned n, float eps) {
  constexpr unsigned WARP_SIZE = 32;
  constexpr unsigned NUM_ROWS_PER_THREAD = NUM_ROWS_PER_WARP;
  constexpr unsigned NUM_COLS_PER_THREAD = N / WARP_SIZE;

  static_assert(N % WARP_SIZE == 0);

  unsigned num_warps = blockDim.x / WARP_SIZE;
  unsigned num_rows_per_block = NUM_ROWS_PER_WARP * num_warps;
  unsigned base_ridx = blockIdx.x * num_rows_per_block;

  unsigned warp_id = threadIdx.x / WARP_SIZE;
  unsigned lane_id = threadIdx.x % WARP_SIZE;

  T elements[NUM_ROWS_PER_THREAD][NUM_COLS_PER_THREAD] = {0.f};
#pragma unroll
  for (unsigned rid = 0; rid < NUM_ROWS_PER_THREAD; ++rid) {
    unsigned ridx = base_ridx + rid + warp_id * NUM_ROWS_PER_THREAD;
    if (ridx >= m) {
      break;
    }
    T* basex = x + ridx * n;
#pragma unroll
    for (unsigned cid = 0; cid < NUM_COLS_PER_THREAD; ++cid) {
      unsigned cidx = cid * WARP_SIZE + lane_id;
      if (cidx < n) {
        elements[rid][cid] = *(basex + cidx);
      }
    }
  }

  float means[NUM_ROWS_PER_THREAD] = {0.f};
#pragma unroll
  for (unsigned rid = 0; rid < NUM_ROWS_PER_THREAD; ++rid) {
    unsigned ridx = base_ridx + rid + warp_id * NUM_ROWS_PER_THREAD;
    if (ridx >= m) {
      break;
    }
#pragma unroll
    for (unsigned cid = 0; cid < NUM_COLS_PER_THREAD; ++cid) {
      means[rid] += convert_to_float(elements[rid][cid]);
    }
#pragma unroll
    for (unsigned s = WARP_SIZE / 2; s >= 1; s /= 2) {
      float otherv = __shfl_xor_sync(0xffffffff, means[rid], s, WARP_SIZE);
      means[rid] += otherv;
    }
    means[rid] /= n;
  }

  // printf("mean: %f\n", means[0]);

  float rstd[NUM_ROWS_PER_THREAD] = {0.f};
#pragma unroll
  for (unsigned rid = 0; rid < NUM_ROWS_PER_THREAD; ++rid) {
    unsigned ridx = base_ridx + rid + warp_id * NUM_ROWS_PER_THREAD;
    if (ridx >= m) {
      break;
    }
#pragma unroll
    for (unsigned cid = 0; cid < NUM_COLS_PER_THREAD; ++cid) {
      unsigned cidx = cid * WARP_SIZE + lane_id;
      if (cidx < n) {
        float v = convert_to_float(elements[rid][cid]);
        v = v - means[rid];
        rstd[rid] += v * v;
      }
    }
#pragma unroll
    for (unsigned s = WARP_SIZE / 2; s >= 1; s /= 2) {
      float otherv = __shfl_xor_sync(0xffffffff, rstd[rid], s, WARP_SIZE);
      rstd[rid] += otherv;
    }
    rstd[rid] /= n;
    rstd[rid] = 1 / sqrt(rstd[rid] + eps);
  }

  // printf("rstd: %f\n", rstd[0]);

#pragma unroll
  for (unsigned rid = 0; rid < NUM_ROWS_PER_THREAD; ++rid) {
    unsigned ridx = base_ridx + rid + warp_id * NUM_ROWS_PER_THREAD;
    if (ridx >= m) {
      break;
    }
    T* baseout = out + ridx * n;
#pragma unroll
    for (unsigned cid = 0; cid < NUM_COLS_PER_THREAD; ++cid) {
      unsigned cidx = cid * WARP_SIZE + lane_id;
      if (cidx < n) {
        float f32w = convert_to_float(w[cidx]);
        float f32b = convert_to_float(b[cidx]);
        float v = (convert_to_float(elements[rid][cid]) - means[rid]) * rstd[rid];
        v = v * f32w + f32b;
        *(baseout + cidx) = convert_from_float<T>(v);
      }
    }
  }
}

template <typename T>
struct LayernormV1 {
  static void impl(T* x, T* out, T* w, T* b, unsigned m, unsigned n, float eps) {
    unsigned grid = m;
    unsigned block = (std::min<unsigned>(n, 1024) + 31) / 32 * 32;
    unsigned smem_size = block / 32;
    auto stream = at::cuda::getCurrentCUDAStream();
    layernorm_kernel_v1<T><<<grid, block, smem_size, stream>>>(x, out, w, b, m, n, eps);
  }
};

template <typename T>
struct LayernormV2 {
  static void impl(T* x, T* out, T* w, T* b, unsigned m, unsigned n, float eps) {
    unsigned grid = m;
    unsigned block = (std::min<unsigned>(n, 1024) + 31) / 32 * 32;
    unsigned smem_size = block / 32;
    auto stream = at::cuda::getCurrentCUDAStream();
    layernorm_kernel_v2<T><<<grid, block, smem_size, stream>>>(x, out, w, b, m, n, eps);
  }
};

template <typename T>
struct LayernormV3 {
  static void impl(T* x, T* out, T* w, T* b, unsigned m, unsigned n, float eps) {
    auto stream = at::cuda::getCurrentCUDAStream();

    unsigned n_2 = 32;
    while (n_2 < n) {
      n_2 *= 2;
    }

    switch (n_2) {
#define LAUNCH_KERNEL(N)                                                                               \
  case N: {                                                                                            \
    unsigned block = 128;                                                                              \
    constexpr unsigned NUM_ROWS_PER_WARP = 1;                                                          \
    const unsigned grid = m / NUM_ROWS_PER_WARP;                                                       \
    layernorm_kernel_v3<T, NUM_ROWS_PER_WARP, N><<<grid, block, 0, stream>>>(x, out, w, b, m, n, eps); \
    break;                                                                                             \
  }

      LAUNCH_KERNEL(32);
      LAUNCH_KERNEL(64);
      LAUNCH_KERNEL(128);
      LAUNCH_KERNEL(256);
      LAUNCH_KERNEL(512);
      LAUNCH_KERNEL(1024);
      LAUNCH_KERNEL(2048);
      LAUNCH_KERNEL(4096);

#undef LAUNCH_KERNEL

      default: {
        unsigned block = 1024;
        unsigned grid = m;
        unsigned smem_size = block / 32;
        layernorm_kernel_v1<T><<<grid, block, smem_size, stream>>>(x, out, w, b, m, n, eps);
      }
    }
  }
};

template <template <class> class KernelLauncher>
void layernorm(torch::Tensor& x, torch::Tensor& out, torch::Tensor& w, torch::Tensor& b, float eps) {
  cr::cr_assert(x.sizes().size() == 2, "only support 2d input.");
  unsigned m = x.size(0);
  unsigned n = x.size(1);
  auto scalar_type = x.scalar_type();
  if (scalar_type == torch::ScalarType::Float) {
    using T = float;
    auto x_ptr = static_cast<T*>(x.data_ptr());
    auto out_ptr = static_cast<T*>(out.data_ptr());
    auto w_ptr = static_cast<T*>(w.data_ptr());
    auto b_ptr = static_cast<T*>(b.data_ptr());
    KernelLauncher<T>::impl(x_ptr, out_ptr, w_ptr, b_ptr, m, n, eps);
  } else if (scalar_type == torch::ScalarType::Half) {
    using T = __half;
    auto x_ptr = static_cast<T*>(x.data_ptr());
    auto out_ptr = static_cast<T*>(out.data_ptr());
    auto w_ptr = static_cast<T*>(w.data_ptr());
    auto b_ptr = static_cast<T*>(b.data_ptr());
    KernelLauncher<T>::impl(x_ptr, out_ptr, w_ptr, b_ptr, m, n, eps);
  } else if (scalar_type == torch::ScalarType::BFloat16) {
    using T = __nv_bfloat16;
    auto x_ptr = static_cast<T*>(x.data_ptr());
    auto out_ptr = static_cast<T*>(out.data_ptr());
    auto w_ptr = static_cast<T*>(w.data_ptr());
    auto b_ptr = static_cast<T*>(b.data_ptr());
    KernelLauncher<T>::impl(x_ptr, out_ptr, w_ptr, b_ptr, m, n, eps);
  } else {
    throw std::runtime_error("not supported type.");
  }

  auto r = cudaGetLastError();
  if (r != cudaSuccess) {
    throw std::runtime_error("cuda error. ret = " + std::to_string(r));
  }
}

static cr::Register _([](pybind11::module& m) {
  m.def("layernorm_v1", &layernorm<LayernormV1>);
  m.def("layernorm_v2", &layernorm<LayernormV2>);
  m.def("layernorm_v3", &layernorm<LayernormV3>);
});

}  // namespace