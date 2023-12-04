#include <vector>
#include "cudnn.h"
#include "stdio.h"

#include "exceptions.h"

// Use global memory.
__global__ void softmax_v1_kernel(float* src, float* dst, int numel) {
  unsigned tid = threadIdx.x;
  __shared__ float* exp_values;
  if (tid == 0) {
    exp_values = static_cast<float*>(malloc(numel * sizeof(float)));
  }
  __syncthreads();
  if (exp_values == nullptr) {
    return;
  }
  for (unsigned i = tid; i < numel; i += blockDim.x) {
    exp_values[i] = expf(src[i]);
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      exp_values[tid] += exp_values[tid + s];
    }
    __syncthreads();
  }
  for (unsigned i = tid; i < numel; i += blockDim.x) {
    dst[i] = expf(src[i]) / exp_values[0];
  }
}

// Use shared memory.
__global__ void softmax_v2_kernel(float* src, float* dst, int numel) {
  extern __shared__ float exp_cache[];
  unsigned tid = threadIdx.x;
  exp_cache[tid] = 0.f;
  for (unsigned i = tid; i < numel; i += blockDim.x) {
    exp_cache[tid] += expf(src[i]);
  }
  __syncthreads();
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
      exp_cache[tid] += exp_cache[tid + s];
    }
    __syncthreads();
  }
  for (unsigned i = tid; i < numel; i += blockDim.x) {
    dst[i] = expf(src[i]) / exp_cache[0];
  }
}

void softmax_v3_kernel(float* src, float* dst, int numel) {
  cudnnHandle_t cudnn_handle;
  {
    cudnnStatus_t r = cudnnCreate(&cudnn_handle);
    if (r != CUDNN_STATUS_SUCCESS) {
      throw CUDNNRuntimeError(r);
    }
  }
  cudnnTensorDescriptor_t desc;
  {
    cudnnStatus_t r = cudnnCreateTensorDescriptor(&desc);
    if (r != CUDNN_STATUS_SUCCESS) {
      throw CUDNNRuntimeError(r);
    }
    r = cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, numel, 1, 1);
    if (r != CUDNN_STATUS_SUCCESS) {
      throw CUDNNRuntimeError(r);
    }
  }

  {
    float alpha = 1.0, beta = 0.0;
    cudnnStatus_t r = cudnnSoftmaxForward(
        cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc, src, &beta, desc, dst);
    if (r != CUDNN_STATUS_SUCCESS) {
      throw CUDNNRuntimeError(r);
    }
  }
}

template <typename F>
std::vector<float> softmax_internal(const std::vector<float>& src, F kernel) {
  float* cuda_src;
  {
    cudaError_t ret = cudaMalloc(&cuda_src, src.size() * sizeof(float));
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }
  {
    cudaError_t ret = cudaMemcpy(cuda_src, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }
  float* cuda_dst;
  {
    cudaError_t ret = cudaMalloc(&cuda_dst, src.size() * sizeof(float));
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }
  cudaStream_t cuda_stream;
  {
    cudaError_t ret = cudaStreamCreate(&cuda_stream);
    if (ret != 0) {
      throw CUDARuntimeError(ret);
    }
  }

  kernel(cuda_src, cuda_dst, src.size(), cuda_stream);

  {
    cudaError_t ret = cudaStreamSynchronize(cuda_stream);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }
  std::vector<float> dst(src.size());
  {
    cudaError_t ret = cudaMemcpy(dst.data(), cuda_dst, src.size() * sizeof(float), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }
  return std::move(dst);
}

std::vector<float> softmax_v1(const std::vector<float>& src) {
  return softmax_internal(src, [](float* src, float* dst, int numel, cudaStream_t cuda_stream) {
    softmax_v1_kernel<<<1, numel, numel * sizeof(float), cuda_stream>>>(src, dst, numel);
  });
}

std::vector<float> softmax_v2(const std::vector<float>& src) {
  return softmax_internal(src, [](float* src, float* dst, int numel, cudaStream_t cuda_stream) {
    softmax_v2_kernel<<<1, numel, numel * sizeof(float), cuda_stream>>>(src, dst, numel);
  });
}

std::vector<float> softmax_v3(const std::vector<float>& src) {
  return softmax_internal(
      src, [](float* src, float* dst, int numel, cudaStream_t cuda_stream) { softmax_v3_kernel(src, dst, numel); });
}