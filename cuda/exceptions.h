#include <exception>

#include <iostream>
#include <string>

#include "cublas_v2.h"
#include "cuda.h"
#include "cudnn.h"

namespace cr {

class CUDADriverError : public std::exception {
 public:
  CUDADriverError(CUresult status, const char* err_msg) {
    err_msg_.append(err_msg).append(", cuda err msg: ");
    cuGetErrorString(status, &err_msg);
    err_msg_.append(err_msg);
  }
  CUDADriverError(CUresult status) : CUDADriverError(status, "") {}

  const char* what() const noexcept override { return err_msg_.c_str(); }

 private:
  std::string err_msg_;
};

class CUDARuntimeError : public std::exception {
 public:
  CUDARuntimeError(cudaError_t status, const char* err_msg) {
    err_msg_.append(err_msg).append(", cuda err msg: ").append(cudaGetErrorString(status));
  }
  CUDARuntimeError(cudaError_t status) : CUDARuntimeError(status, "") {}

  const char* what() const noexcept override { return err_msg_.c_str(); }

 private:
  std::string err_msg_;
};

class CUBLASRuntimeError : public std::exception {
 public:
  CUBLASRuntimeError(cublasStatus_t status, const char* err_msg) {
    err_msg_.append(err_msg).append(", cuda err msg: ").append(cublasGetErrorString(status));
  }
  CUBLASRuntimeError(cublasStatus_t status) : CUBLASRuntimeError(status, "") {}

  const char* what() const noexcept override { return err_msg_.c_str(); }

 private:
  const char* cublasGetErrorString(cublasStatus_t status) const noexcept {
    switch (status) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
      default:
        return "Unknown cuBLAS status";
    }
  }

 private:
  std::string err_msg_;
};

class CUDNNRuntimeError : public std::exception {
 public:
  CUDNNRuntimeError(cudnnStatus_t status, const char* err_msg) {
    err_msg_.append(err_msg).append(", cuda err msg: ").append(cudnnGetErrorString(status));
  }
  CUDNNRuntimeError(cudnnStatus_t status) : CUDNNRuntimeError(status, "") {}

  const char* what() const noexcept override { return err_msg_.c_str(); };

 private:
  std::string err_msg_;
};

template <typename T>
inline void check_cuda_err(T, const char* msg);

template <typename T>
inline void check_cuda_err(T err_code) {
  return check_cuda_err(err_code, "");
}

template <>
inline void check_cuda_err<CUresult>(CUresult ret, const char* msg) {
  if (ret != CUDA_SUCCESS) {
    throw CUDADriverError(ret, msg);
  }
}

template <>
inline void check_cuda_err<cudaError_t>(cudaError_t ret, const char* msg) {
  if (ret != cudaSuccess) {
    throw CUDARuntimeError(ret, msg);
  }
}

template <>
inline void check_cuda_err<cublasStatus_t>(cublasStatus_t ret, const char* msg) {
  if (ret != CUBLAS_STATUS_SUCCESS) {
    throw CUBLASRuntimeError(ret, msg);
  }
}

template <>
inline void check_cuda_err<cudnnStatus_t>(cudnnStatus_t ret, const char* msg) {
  if (ret != CUDNN_STATUS_SUCCESS) {
    throw CUDNNRuntimeError(ret, msg);
  }
}

}  // namespace cr
