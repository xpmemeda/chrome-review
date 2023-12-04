#include "kernels/softmax.h"

#include <numeric>
#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAGuard.h"
#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cudnn.h"

#include "support/assert.h"
#include "support/exceptions.h"

namespace cr {

namespace {

template <unsigned int N>
struct is_power_of_two {
  static constexpr bool value = !(N & (N - 1));
};

template <typename T>
struct Add {
  static constexpr T init_value = T{};
  static __device__ __forceinline__ T apply(T a, T b) { return a + b; }
};

template <typename T>
struct Max {
  static constexpr T init_value =
      std::is_integral_v<T> ? std::numeric_limits<T>::min() : -std::numeric_limits<T>::infinity();
  static __device__ __forceinline__ T apply(T a, T b) { return a > b ? a : b; }
};

template <template <typename> typename ReduceOp, unsigned WARP_SIZE>
inline __device__ float cr_reduce(float* smem, float value) {
  unsigned warp = threadIdx.x / WARP_SIZE;
  unsigned lane = threadIdx.x % WARP_SIZE;
  unsigned num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

  // Reduce inside warp.
  for (unsigned s = 16; s >= 1; s /= 2) {
    auto ex_value = __shfl_xor_sync(0xffffffff, value, s, WARP_SIZE);
    value = ReduceOp<float>::apply(value, ex_value);
  }

  // Exchange value by smem with other warps.
  if (lane == 0) {
    smem[warp] = value;
  }
  __syncthreads();
  if (lane < num_warps) {
    value = smem[lane];
  } else {
    value = ReduceOp<float>::init_value;
  }
  __syncthreads();

  // Reduce inside warp
  for (unsigned s = 16; s >= 1; s /= 2) {
    auto x = __shfl_xor_sync(0xffffffff, value, s, WARP_SIZE);
    value = ReduceOp<float>::apply(value, x);
  }

  return value;
}

// Naive.
__global__ void softmax_v1_kernel(float* src, float* dst, unsigned rows, unsigned cols) {
  unsigned block_offset = blockIdx.x * cols;
  src = src + block_offset;
  dst = dst + block_offset;

  unsigned numel = cols;
  unsigned tid = threadIdx.x;

  __shared__ float* tmp_exp;
  if (tid == 0) {
    tmp_exp = static_cast<float*>(malloc(numel * sizeof(float)));
  }
  __syncthreads();
  for (unsigned i = tid; i < numel; i += blockDim.x) {
    tmp_exp[i] = expf(src[i]);
  }
  __syncthreads();
  for (unsigned int s = 1; s < numel; s *= 2) {
    for (unsigned task_id = tid; task_id < numel; task_id += blockDim.x) {
      if (task_id % (2 * s) == 0 && task_id + s < numel) {
        tmp_exp[task_id] += tmp_exp[task_id + s];
      }
    }
    __syncthreads();
  }
  for (unsigned i = tid; i < numel; i += blockDim.x) {
    dst[i] = expf(src[i]) / tmp_exp[0];
  }
  if (tid == 0) {
    free(tmp_exp);
  }
}

// cuDNN.
void softmax_v2_kernel(float* src, float* dst, unsigned rows, unsigned cols, const cudaStream_t stream) {
  cudnnHandle_t cudnn_handle;
  {
    cudnnStatus_t r = cudnnCreate(&cudnn_handle);
    if (r != CUDNN_STATUS_SUCCESS) {
      // throw CUDNNRuntimeError(r);
    }
  }
  {
    cudnnStatus_t r = cudnnSetStream(cudnn_handle, stream);
    if (r != CUDNN_STATUS_SUCCESS) {
      // throw CUDNNRuntimeError(r);
    }
  }
  cudnnTensorDescriptor_t desc;
  {
    cudnnStatus_t r = cudnnCreateTensorDescriptor(&desc);
    if (r != CUDNN_STATUS_SUCCESS) {
      // throw CUDNNRuntimeError(r);
    }
    r = cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, rows, cols, 1, 1);
    if (r != CUDNN_STATUS_SUCCESS) {
      // throw CUDNNRuntimeError(r);
    }
  }

  {
    float alpha = 1.0, beta = 0.0;
    cudnnStatus_t r = cudnnSoftmaxForward(
        cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, desc, src, &beta, desc, dst);
    if (r != CUDNN_STATUS_SUCCESS) {
      // throw CUDNNRuntimeError(r);
    }
  }
}

// Brief: One "BLOCK" works on one "ROW", use "SHARED_MEMORY" to exchange data between "WARP"s
__global__ void softmax_v3_kernel(float* src, float* dst, unsigned rows, unsigned cols) {
  assert(blockDim.x % 32 == 0);

  unsigned block_offset = blockIdx.x * cols;
  src = src + block_offset;
  dst = dst + block_offset;

  unsigned tid = threadIdx.x;

  // Online softmax.
  float local_expsum = 0.f, local_max = -std::numeric_limits<float>::infinity();
  for (unsigned i = tid; i < cols; i += blockDim.x) {
    float v = src[i];
    float new_local_max = std::max<float>(v, local_max);
    local_expsum = local_expsum * std::exp(local_max - new_local_max) + std::exp(v - new_local_max);
    local_max = new_local_max;
  }

  __shared__ float smem[32];

  float global_max = cr_reduce<Max, 32>(smem, local_max);
  local_expsum = local_expsum * std::exp(local_max - global_max);
  local_expsum = cr_reduce<Add, 32>(smem, local_expsum);

  for (unsigned i = tid; i < cols; i += blockDim.x) {
    dst[i] = std::exp(src[i] - global_max) / local_expsum;
  }
}

// Brief: One "WARP" works on one "ROW", not use "batch load" and "batch stort", this is faster than batched version.
//
// **Important**
// Maximum number of threads per block: 1024.
// Number of 32-bit regular registers per multiprocessor: 64K.
// Maximum number of 32-bit regular registers per thread: 256.
//
// **Not important**
// Maximum number of resident blocks per multiprocessor: 16 ~ 32.
// Maximum number of resident warps per multiprocessor: 32 ~ 64.
template <unsigned WARP_SIZE, unsigned NUM_WARPS, unsigned ROWS_PER_WARP, unsigned NUM_ELEMENTS /*must be pow of two*/>
__global__ void softmax_v4_kernel(float* src, float* dst, unsigned rows, unsigned cols) {
  // check NUM_ELEMENTS is pow of two.
  static_assert(is_power_of_two<NUM_ELEMENTS>::value);
  static_assert(NUM_ELEMENTS % WARP_SIZE == 0);
  constexpr unsigned COLS_PER_THREAD = NUM_ELEMENTS / WARP_SIZE;

  unsigned tid = threadIdx.x;
  unsigned warp_id = tid / WARP_SIZE;
  unsigned lane_id = tid % WARP_SIZE;

  unsigned row_idx = blockIdx.x * NUM_WARPS * ROWS_PER_WARP + warp_id;
  if (row_idx >= rows) {
    return;
  }
  src = src + row_idx * cols + lane_id;
  dst = dst + row_idx * cols + lane_id;

  // Load src.
  float local_elements[ROWS_PER_WARP][COLS_PER_THREAD];
#pragma unroll
  for (unsigned row_idx = 0; row_idx < ROWS_PER_WARP; ++row_idx) {
#pragma unroll
    for (unsigned it = 0; it < COLS_PER_THREAD; ++it) {
      unsigned element_idx = lane_id + it * WARP_SIZE;
      if (element_idx < cols) {
        local_elements[row_idx][it] = src[row_idx * cols + it * WARP_SIZE];
      } else {
        local_elements[row_idx][it] = -std::numeric_limits<float>::infinity();
      }
    }
  }

  // Calc max.
  float local_max[ROWS_PER_WARP]{-std::numeric_limits<float>::infinity()};
#pragma unroll
  for (unsigned row_idx = 0; row_idx < ROWS_PER_WARP; ++row_idx) {
#pragma unroll
    for (unsigned i = 0; i < COLS_PER_THREAD; ++i) {
      local_max[row_idx] = std::max<float>(local_max[row_idx], local_elements[row_idx][i]);
    }
  }
#pragma unroll
  for (unsigned row_idx = 0; row_idx < ROWS_PER_WARP; ++row_idx) {
#pragma unroll
    for (unsigned s = 16; s >= 1; s /= 2) {
      local_max[row_idx] = std::max<float>(local_max[row_idx], __shfl_xor_sync(0xffffffff, local_max[row_idx], s, 32));
    }
  }

  // Calc exp.
  float local_expsum[ROWS_PER_WARP]{0.0f};
#pragma unroll
  for (unsigned row_idx = 0; row_idx < ROWS_PER_WARP; ++row_idx) {
#pragma unroll
    for (unsigned i = 0; i < COLS_PER_THREAD; ++i) {
      // __expf is only slightly faster than expf: 12.78 -> 12.51
      local_elements[row_idx][i] = __expf(local_elements[row_idx][i] - local_max[row_idx]);
      local_expsum[row_idx] += local_elements[row_idx][i];
    }
  }
#pragma unroll
  for (unsigned row_idx = 0; row_idx < ROWS_PER_WARP; ++row_idx) {
#pragma unroll
    for (unsigned s = 16; s >= 1; s /= 2) {
      // mask = 0xffffffff, which means all lanes will be waited and gathered.
      local_expsum[row_idx] += __shfl_xor_sync(0xffffffff, local_expsum[row_idx], s, 32);
    }
  }

  // Save dst.
#pragma unroll
  for (unsigned row_idx = 0; row_idx < ROWS_PER_WARP; ++row_idx) {
    for (unsigned it = 0; it < COLS_PER_THREAD; ++it) {
      unsigned element_idx = lane_id + it * WARP_SIZE;
      if (element_idx < cols) {
        dst[row_idx * cols + it * WARP_SIZE] = local_elements[row_idx][it] / local_expsum[row_idx];
      } else {
        break;
      }
    }
  }
}

}  // namespace

void softmax_v1(torch::Tensor& x, torch::Tensor& out) {
  unsigned rank = x.sizes().size();
  cr::cr_assert(rank >= 1, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(x.is_cuda(), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(out.is_cuda(), "%s:%d", __FILE__, __LINE__);
  const cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream();
  unsigned cols = x.size(rank - 1);
  unsigned rows = 1;
  for (unsigned i = 0; i < rank - 1; ++i) {
    rows *= x.size(i);
  }

  float* x_ptr = reinterpret_cast<float*>(x.data_ptr());
  float* out_ptr = reinterpret_cast<float*>(out.data_ptr());
  softmax_v1_kernel<<<rows, std::min<unsigned>(cols, 1024), 0, torch_stream>>>(x_ptr, out_ptr, rows, cols);
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw CUDARuntimeError(ret);
  }
}

void softmax_v2(torch::Tensor& x, torch::Tensor& out) {
  unsigned rank = x.sizes().size();
  cr::cr_assert(rank >= 1, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(x.is_cuda(), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(out.is_cuda(), "%s:%d", __FILE__, __LINE__);
  const cudaStream_t torch_stream = at::cuda::getCurrentCUDAStream();
  unsigned cols = x.size(rank - 1);
  unsigned rows = 1;
  for (unsigned i = 0; i < rank - 1; ++i) {
    rows *= x.size(i);
  }
  softmax_v2_kernel(
      reinterpret_cast<float*>(x.data_ptr()), reinterpret_cast<float*>(out.data_ptr()), rows, cols, torch_stream);
}

void softmax_v3(torch::Tensor& x, torch::Tensor& out) {
  unsigned rank = x.sizes().size();
  cr::cr_assert(rank >= 1, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(x.is_cuda(), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(out.is_cuda(), "%s:%d", __FILE__, __LINE__);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  unsigned cols = x.size(rank - 1);
  unsigned rows = 1;
  for (unsigned i = 0; i < rank - 1; ++i) {
    rows *= x.size(i);
  }

  float* x_ptr = reinterpret_cast<float*>(x.data_ptr());
  float* out_ptr = reinterpret_cast<float*>(out.data_ptr());
  unsigned num_blocks = rows;
  unsigned num_threads = (std::min<unsigned>(cols, 1024) + 31) / 32 * 32;  // Align to 32.
  softmax_v3_kernel<<<num_blocks, num_threads, 0, stream>>>(x_ptr, out_ptr, rows, cols);
  cudaError_t ret = cudaGetLastError();
  if (ret != cudaSuccess) {
    throw CUDARuntimeError(ret);
  }
}

void softmax_v4(torch::Tensor& x, torch::Tensor& out) {
  unsigned rank = x.sizes().size();
  cr::cr_assert(rank >= 1, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(x.is_cuda(), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(out.is_cuda(), "%s:%d", __FILE__, __LINE__);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  unsigned cols = x.size(rank - 1);
  unsigned rows = 1;
  for (unsigned i = 0; i < rank - 1; ++i) {
    rows *= x.size(i);
  }

  float* x_ptr = reinterpret_cast<float*>(x.data_ptr());
  float* out_ptr = reinterpret_cast<float*>(out.data_ptr());

  unsigned num_elements = 32;
  while (num_elements < cols) {
    num_elements *= 2;
  }

  // Torch hardcode 128 threads.
  constexpr unsigned WARP_SIZE = 32;
  constexpr unsigned NUM_WARPS = 4;
  constexpr unsigned ROWS_PER_WARP = 1;
  unsigned num_blocks = (rows + NUM_WARPS * ROWS_PER_WARP - 1) / NUM_WARPS / ROWS_PER_WARP;
  unsigned num_threads = WARP_SIZE * NUM_WARPS;

  switch (num_elements) {
#define LAUNCH_SOFTMAX_KERNEL(NUM_ELEMENTS)                                   \
  case NUM_ELEMENTS: {                                                        \
    softmax_v4_kernel<WARP_SIZE, NUM_WARPS, ROWS_PER_WARP, NUM_ELEMENTS>      \
        <<<num_blocks, num_threads, 0, stream>>>(x_ptr, out_ptr, rows, cols); \
    break;                                                                    \
  }
    LAUNCH_SOFTMAX_KERNEL(32);
    LAUNCH_SOFTMAX_KERNEL(64);
    LAUNCH_SOFTMAX_KERNEL(128);
    LAUNCH_SOFTMAX_KERNEL(256);
    LAUNCH_SOFTMAX_KERNEL(512);
    LAUNCH_SOFTMAX_KERNEL(1024);
    default: {
      break;
    }
  }
}

}  // namespace cr
