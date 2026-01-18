#include <ATen/Tensor.h>
#include <cuda.h>
#include <cudaProfiler.h>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <vector>

#include "./module.h"

namespace {

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    CUresult err = (call);                                                   \
    if (err != CUDA_SUCCESS) {                                               \
      const char* msg;                                                       \
      cuGetErrorString(err, &msg);                                           \
      std::cerr << "CUDA Error: " << msg << " at " << __LINE__ << std::endl; \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

__global__ void compute_kernel(float* ptr, size_t N) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    float x = ptr[idx];
    x = sinf(x) + cosf(x);
    ptr[idx] = x;
  }
}

class GreenContextManager {
 public:
  GreenContextManager(int n) : dev_resources_groups_(n), gctxs_(n), ctxs_(n) {
    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));

    int smCount = 0;
    CHECK_CUDA(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
    std::cout << "Device has " << smCount << " SMs\n";

    CUdevResource dev_resources;
    CHECK_CUDA(cuDeviceGetDevResource(dev, &dev_resources, CU_DEV_RESOURCE_TYPE_SM));
    std::cout << "Got " << dev_resources.sm.smCount << " SM resources.\n";

    unsigned min_sm_count = smCount / n;
    unsigned ngroups = n;
    for (; min_sm_count > 0; --min_sm_count) {
      cuDevSmResourceSplitByCount(
          NULL, &ngroups, &dev_resources, NULL, CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_count);
      if (ngroups >= n) {
        break;
      }
    }

    CUdevResource rem_resources;
    CHECK_CUDA(cuDevSmResourceSplitByCount(dev_resources_groups_.data(), &ngroups, &dev_resources, &rem_resources,
        CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING, min_sm_count));
    printf("Split SM resources: n = %d, ngroups = %u\n", n, ngroups);
    for (int i = 0; i < ngroups; ++i) {
      printf("Group %d Occupy %u SMs.\n", i, dev_resources_groups_[i].sm.smCount);
    }
    printf("Rem %u SMs.\n", rem_resources.sm.smCount);

    for (int i = 0; i < n; ++i) {
      CUdevResourceDesc resource_desc;
      CHECK_CUDA(cuDevResourceGenerateDesc(&resource_desc, &dev_resources_groups_[i], 1));
      CHECK_CUDA(cuGreenCtxCreate(&gctxs_[i], resource_desc, dev, CU_GREEN_CTX_DEFAULT_STREAM));
      CHECK_CUDA(cuCtxFromGreenCtx(&ctxs_[i], gctxs_[i]));
    }
  }

  CUcontext get(int index) const { return ctxs_[index]; }

  ~GreenContextManager() {
    for (auto gctx : gctxs_) {
      CHECK_CUDA(cuGreenCtxDestroy(gctx));
    }
  }

 private:
  std::vector<CUdevResource> dev_resources_groups_;
  std::vector<CUgreenCtx> gctxs_;
  std::vector<CUcontext> ctxs_;
};

class ContextManager {
 public:
  ContextManager(int n) : ctxs_(n) {
    CHECK_CUDA(cuInit(0));
    CUdevice dev;
    CHECK_CUDA(cuDeviceGet(&dev, 0));

    for (int i = 0; i < n; ++i) {
      CHECK_CUDA(cuCtxCreate(&ctxs_[i], 0, dev));
    }
  }

  CUcontext get(int index) const { return ctxs_[index]; }

  ~ContextManager() {
    for (auto ctx : ctxs_) {
      cuCtxDestroy(ctx);
    }
  }

 private:
  std::vector<CUcontext> ctxs_;
};

void run_kernel_in_green_ctx(float* ptr, size_t N) {
  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  compute_kernel<<<blocks, threads>>>(ptr, N);
}

class ThreadWorker {
 public:
  ThreadWorker(CUcontext ctx) : running_(true) { worker_ = std::thread(&ThreadWorker::loop, this, ctx); }
  ~ThreadWorker() {
    submit([this]() { running_.store(false); });
    worker_.join();
  }

  void submit(std::function<void()> task) {
    {
      std::lock_guard guard(m_);
      tasks_.push(task);
    }
    cv_.notify_one();
  }

  void sync() {
    std::atomic done = false;

    submit([&]() { done.store(true); });

    while (!done.load()) {
    }
    return;
  }

 private:
  void loop(CUcontext ctx) {
    CHECK_CUDA(cuCtxPushCurrent(ctx));

    std::stringstream oss;
    oss << std::this_thread::get_id();

    uint64_t thread_id = std::stoull(oss.str());

    printf("ThreadWorker %" PRIu64 "(%p) use Ctx %p\n", thread_id, reinterpret_cast<void*>(thread_id), ctx);

    while (running_) {
      std::function<void()> task;

      {
        std::unique_lock lk(m_);
        cv_.wait(lk, [this]() { return !running_ || !tasks_.empty(); });
        if (!running_) {
          break;
        }
        task = tasks_.front();
      }

      task();

      {
        std::lock_guard guard(m_);
        tasks_.pop();
      }
    }

    printf("ThreadWorker %" PRIu64 " (%p) finished.\n", thread_id, reinterpret_cast<void*>(thread_id));
  }

 private:
  std::atomic<bool> running_;
  std::thread worker_;

  std::mutex m_;
  std::queue<std::function<void()>> tasks_;
  std::condition_variable cv_;
};

void greenctx_y(at::Tensor& x, at::Tensor& y) {
  static GreenContextManager gcm(2);
  static ThreadWorker worker_a(gcm.get(0));
  static ThreadWorker worker_b(gcm.get(1));

  worker_a.submit([&]() { run_kernel_in_green_ctx(static_cast<float*>(x.data_ptr()), x.numel()); });
  worker_b.submit([&]() { run_kernel_in_green_ctx(static_cast<float*>(y.data_ptr()), y.numel()); });
  worker_a.sync();
  worker_b.sync();
}

void greenctx_n(at::Tensor& x, at::Tensor& y) {
  static ContextManager gcm(2);
  static ThreadWorker worker_a(gcm.get(0));
  static ThreadWorker worker_b(gcm.get(1));

  worker_a.submit([&]() { run_kernel_in_green_ctx(static_cast<float*>(x.data_ptr()), x.numel()); });
  worker_b.submit([&]() { run_kernel_in_green_ctx(static_cast<float*>(y.data_ptr()), y.numel()); });
  worker_a.sync();
  worker_b.sync();
}

static cr::Register _([](pybind11::module& m) {
  m.def("greenctx_y", &greenctx_y);
  m.def("greenctx_n", &greenctx_n);
});

}  // namespace
