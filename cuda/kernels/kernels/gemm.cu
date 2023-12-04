#include "kernels/gemm.h"

#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAStream.h"
#include "cublas_v2.h"

#include "support/assert.h"
#include "support/exceptions.h"

namespace cr {

namespace {

void gemm_kernel_v1(float* a, float* b, float* c, unsigned m, unsigned n, unsigned k, cudaStream_t stream) {
  cublasHandle_t cublas_handle;
  {
    cublasStatus_t r = cublasCreate(&cublas_handle);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    cublasStatus_t r = cublasSetStream(cublas_handle, stream);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    float alpha = 1.f, beta = 0.f;
    // CUDA use column major matrix, so we actually need C(t).
    // C(t) = B(t)A(t)
    cublasStatus_t r = cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &beta, c, n);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    cublasStatus_t r = cublasDestroy(cublas_handle);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
}

void gemm_kernel_v2(float* a, float* b, float* c, unsigned m, unsigned n, unsigned k, cudaStream_t stream) {
  cublasHandle_t cublas_handle;
  {
    cublasStatus_t r = cublasCreate(&cublas_handle);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    cublasStatus_t r = cublasSetStream(cublas_handle, stream);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    cublasStatus_t r = cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    float alpha = 1.f, beta = 0.f;
    // Tensor core.
    cublasStatus_t r = cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, CUDA_R_32F, n, a,
        CUDA_R_32F, k, &beta, c, CUDA_R_32F, n, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
  {
    cublasStatus_t r = cublasDestroy(cublas_handle);
    if (r != CUBLAS_STATUS_SUCCESS) {
      throw CUBLASRuntimeError(r);
    }
  }
}

}  // namespace

void gemm_v1(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c) {
  cr::cr_assert(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(a.size(0) == c.size(0), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(b.size(1) == c.size(1), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(a.size(1) == b.size(0), "%s:%d", __FILE__, __LINE__);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  float* a_ptr = reinterpret_cast<float*>(a.data_ptr());
  float* b_ptr = reinterpret_cast<float*>(b.data_ptr());
  float* c_ptr = reinterpret_cast<float*>(c.data_ptr());
  unsigned m = a.size(0), k = a.size(1), n = b.size(1);
  gemm_kernel_v1(a_ptr, b_ptr, c_ptr, m, n, k, stream);
}

void gemm_v2(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c) {
  cr::cr_assert(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(a.size(0) == c.size(0), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(b.size(1) == c.size(1), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(a.size(1) == b.size(0), "%s:%d", __FILE__, __LINE__);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  float* a_ptr = reinterpret_cast<float*>(a.data_ptr());
  float* b_ptr = reinterpret_cast<float*>(b.data_ptr());
  float* c_ptr = reinterpret_cast<float*>(c.data_ptr());
  unsigned m = a.size(0), k = a.size(1), n = b.size(1);
  gemm_kernel_v2(a_ptr, b_ptr, c_ptr, m, n, k, stream);
}

}  // namespace cr
