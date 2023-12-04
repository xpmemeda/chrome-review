#include <exception>

class CUDARuntimeError : public std::exception {
  cudaError_t _status;

 public:
  CUDARuntimeError(cudaError_t status) : _status(status) {}

  const char* what() const noexcept override { return cudaGetErrorString(_status); }
};

class CUDNNRuntimeError : public std::exception {
  cudnnStatus_t _status;

 public:
  CUDNNRuntimeError(cudnnStatus_t status) : _status(status) {}

  const char* what() const noexcept override { return cudnnGetErrorString(_status); };
};