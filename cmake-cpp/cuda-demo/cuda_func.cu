#include <cuda_runtime.h>
#include <tuple>
#include <vector>

class CUDARuntimeError : public std::exception {
  cudaError_t _status;

public:
  CUDARuntimeError(cudaError_t status) : _status(status) {}

  const char *what() const noexcept override { return cudaGetErrorString(_status); }
};

std::tuple<size_t, size_t> getSuitableKernelSize(size_t len) {
  static cudaDeviceProp prop;
  cudaError_t ret = cudaGetDeviceProperties(&prop, 0);
  if (ret != cudaSuccess)
    throw CUDARuntimeError(ret);
  size_t _maxThreadPerBlock = prop.maxThreadsPerBlock;
  size_t _maxBlockPerGrimX = prop.maxGridSize[0];
  size_t _maxBlockPerGrimY = prop.maxGridSize[1];
  size_t _maxBlockPerGrimZ = prop.maxGridSize[2];
  printf("%zu, %zu, %zu, %zu\n", _maxThreadPerBlock, _maxBlockPerGrimX, _maxBlockPerGrimY, _maxBlockPerGrimZ);

  size_t blockCount = 1, threadCount = _maxThreadPerBlock;
  if (len < _maxThreadPerBlock)
    threadCount = len;
  else
    blockCount = (len + _maxThreadPerBlock - 1) / _maxThreadPerBlock;
  blockCount = std::min(blockCount, _maxBlockPerGrimX);

  return std::make_tuple(blockCount, threadCount);
}

template <class T> static void __global__ _cuda_zeros(T *a, unsigned total) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    a[i] = static_cast<T>(0);
}

template <class T> static void __global__ _cuda_add_one(T *a, unsigned total) {
  const unsigned tid = threadIdx.x + blockDim.x * blockIdx.x;
  const unsigned skip = blockDim.x * gridDim.x;
  for (unsigned i = tid; i < total; i += skip)
    a[i] += static_cast<T>(1);
}

template <typename T> std::vector<T> ones(size_t num_elem) {
  size_t blockCount, threadCount;
  std::tie(blockCount, threadCount) = getSuitableKernelSize(num_elem);

  T *mem_cuda;
  {
    cudaError_t ret = cudaMalloc(&mem_cuda, num_elem * sizeof(T));
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
  }

  cudaStream_t cuda_stream;
  {
    cudaError_t ret = cudaStreamCreate(&cuda_stream);
    if (ret != 0)
      throw CUDARuntimeError(ret);
  }

  {
    _cuda_zeros<<<blockCount, threadCount, 0, cuda_stream>>>(mem_cuda, num_elem);
    cudaError_t ret = cudaGetLastError();
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
  }

  {
    _cuda_add_one<<<blockCount, threadCount, 0, cuda_stream>>>(mem_cuda, num_elem);
    cudaError_t ret = cudaGetLastError();
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
  }

  {
    cudaError_t ret = cudaStreamSynchronize(cuda_stream);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }

  std::vector<T> mem_host(num_elem);
  {
    cudaError_t ret = cudaMemcpy(mem_host.data(), mem_cuda, num_elem * sizeof(T), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess)
      throw CUDARuntimeError(ret);
  }

  return mem_host;
}

template std::vector<int> ones(size_t num_elem);
template std::vector<double> ones(size_t num_elem);
