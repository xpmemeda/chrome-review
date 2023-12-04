#include <exception>

#include "cublas_v2.h"
#include "cudnn.h"

namespace cr {

class CUDARuntimeError : public std::exception {
  cudaError_t _status;

 public:
  CUDARuntimeError(cudaError_t status) : _status(status) {}

  const char* what() const noexcept override { return cudaGetErrorString(_status); }
};

class CUBLASRuntimeError : public std::exception {
  cublasStatus_t _status;

 public:
  CUBLASRuntimeError(cublasStatus_t status) : _status(status) {}
  const char* what() const noexcept override {
    switch (_status) {
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "The cuBLAS library was not initialized.";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the cuBLAS library.";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a negative vector size, for example).";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device architecture; usually caused by the lack of "
               "support for double precision.";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to execute.";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "An internal cuBLAS operation failed.";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "The functionnality requested is not supported.";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "The functionnality requested requires some license and an error was detected when trying to check the "
               "current licensing.";
      default:
        return "Unknow error.";
    }
  }
};

class CUDNNRuntimeError : public std::exception {
  cudnnStatus_t _status;

 public:
  CUDNNRuntimeError(cudnnStatus_t status) : _status(status) {}

  const char* what() const noexcept override { return cudnnGetErrorString(_status); };
};

template <typename T>
inline void check_cuda_err(T);

template <>
inline void check_cuda_err<cudaError_t>(cudaError_t ret) {
  if (ret != cudaSuccess) {
    throw CUDARuntimeError(ret);
  }
}

template <>
inline void check_cuda_err<cublasStatus_t>(cublasStatus_t ret) {
  if (ret != CUBLAS_STATUS_SUCCESS) {
    throw CUBLASRuntimeError(ret);
  }
}

template <>
inline void check_cuda_err<cudnnStatus_t>(cudnnStatus_t ret) {
  if (ret != CUDNN_STATUS_SUCCESS) {
    throw CUDNNRuntimeError(ret);
  }
}

}  // namespace cr
