#include <pthread.h>
#include <syscall.h>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

int64_t lap(std::chrono::_V2::system_clock::time_point t0, std::chrono::_V2::system_clock::time_point t1) {
  return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

void d2h(at::cuda::CUDAStream at_stream, at::Tensor& a, std::atomic<int>& gun) {
  char name[16];
  pthread_getname_np(pthread_self(), name, sizeof(name));

  at::cuda::CUDAStreamGuard at_stream_guard(at_stream);

  auto z = at::empty_like(a, at::TensorOptions().device(at::kCPU)).pin_memory();

  for (int i = 0; i < 10; ++i) {
    while (gun.load() != i) {
    }

    auto x = at::zeros_like(a);
    auto y = at::empty_like(x, at::TensorOptions().device(at::kCPU));

    auto t0 = std::chrono::system_clock::now();
    z.copy_(x, true);
    at::cuda::getCurrentCUDAStream().synchronize();
    auto t1 = std::chrono::system_clock::now();
    y.copy_(z);
    auto t2 = std::chrono::system_clock::now();

    size_t nbytes = x.nbytes();

    int64_t d2h_t = lap(t0, t1);
    int64_t h2h_t = lap(t1, t2);

    double d2h_throughput = nbytes * 1e-9 / (d2h_t * 1e-3);
    double h2h_throughput = nbytes * 1e-9 / (h2h_t * 1e-3);

    printf("%s: d2h %d : %lld ms, nbytes %f GB, d2h_throughput %f GB/s, h2h_throughput %f GB/s\n", name, i, d2h_t,
        nbytes * 1e-9, d2h_throughput, h2h_throughput);
  }
}

void d2h_cu_stream(at::Tensor& a, std::atomic<int>& gun) {
  class StreamGuard {
   public:
    StreamGuard(cudaStream_t stream) : stream_(stream) { cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking); }
    ~StreamGuard() {
      cudaStreamSynchronize(stream_);
      cudaStreamDestroy(stream_);
    }

   private:
    cudaStream_t stream_;
  };

  cudaStream_t stream;
  StreamGuard stream_guard(stream);

  pthread_setname_np(pthread_self(), "d2h_cu_stream");
  nvtxNameOsThreadA(syscall(SYS_gettid), "d2h_cu_stream");

  d2h(at::cuda::getStreamFromExternal(stream, 0), a, gun);
}

void d2h_at_stream(at::Tensor& a, std::atomic<int>& gun) {
  pthread_setname_np(pthread_self(), "d2h_at_stream");
  nvtxNameOsThreadA(syscall(SYS_gettid), "d2h_at_stream");

  // TODO: Nsight see that d2h run on default cuda stream.
  d2h(at::cuda::getStreamFromPool(false, 0), a, gun);
}

void compute(at::Tensor& x, std::atomic<int>& gun) {
  {
    for (int i = 0; i < 10; ++i) {
      while (gun.load() != i) {
      }

      auto t0 = std::chrono::system_clock::now();
      auto t1 = std::chrono::system_clock::now();

      while (lap(t0, t1) < 500) {
        at::bmm(x, x);
        at::cuda::getCurrentCUDAStream().synchronize();
        t1 = std::chrono::system_clock::now();
      }

      printf("com %d : %lldms\n", i, lap(t0, t1));
    }
  }
}

int main() {
  std::atomic<int> gun = -1;

  // 1G.
  at::Tensor x = at::empty({256, 1024, 1024}, at::TensorOptions().dtype(at::ScalarType::Float).device(at::kCUDA));

  auto d2h_cu_stream_t = std::thread(d2h_cu_stream, std::ref(x), std::ref(gun));
  auto d2h_at_stream_t = std::thread(d2h_at_stream, std::ref(x), std::ref(gun));
  auto com_t = std::thread(compute, std::ref(x), std::ref(gun));

  std::this_thread::sleep_for(std::chrono::seconds(5));
  for (int i = 0; i < 10; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    gun.store(i);
  }

  d2h_cu_stream_t.join();
  d2h_at_stream_t.join();
  com_t.join();

  return 0;
}
