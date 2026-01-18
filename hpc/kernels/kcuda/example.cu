// bank_conflict.cu
#include <cuda_runtime.h>
#include <cstdio>

#ifndef CHECK
#  define CHECK(call)                                                                                       \
    do {                                                                                                    \
      cudaError_t err = call;                                                                               \
      if (err != cudaSuccess) {                                                                             \
        fprintf(stderr, "CUDA error %s (%d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
        exit(1);                                                                                            \
      }                                                                                                     \
    } while (0)
#endif

// 配置参数：可按需调整
constexpr int BLOCK_SIZE = 256;  // 每个 block 的线程数（应为 32 的倍数）
constexpr int GRID_SIZE = 256;   // block 个数，保证占满 GPU
constexpr int REPEAT = 4096;     // 循环重复次数，用来放大差异

// 无冲突：每个线程访问不同 bank（stride=1）
__global__ void kernel_no_conflict(float* out) {
  __shared__ volatile float s[BLOCK_SIZE * 1];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  int idx = tid * 1;
  s[idx] = float(tid);
  __syncthreads();

  float acc = 0.f;
  for (int it = 0; it < REPEAT; ++it) {
    acc += s[idx];
    s[idx] = acc;
  }
  out[gtid] = acc;
}

// 轻度冲突：stride=2 会造成同一 warp 内 (t, t+16) 落到同一 bank（2 路冲突）
__global__ void kernel_conflict_stride2(float* out) {
  __shared__ volatile float s[BLOCK_SIZE * 2];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  int idx = tid * 2;
  s[idx] = float(tid);
  __syncthreads();

  float acc = 0.f;
  for (int it = 0; it < REPEAT; ++it) {
    acc += s[idx];
    s[idx] = acc;
  }
  out[gtid] = acc;
}

// 严重冲突：stride=16 在 32 个 bank 中只用到两个 bank（16 路冲突）
__global__ void kernel_conflict_stride16(float* out) {
  __shared__ volatile float s[BLOCK_SIZE * 16];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  int idx = tid * 16;
  s[idx] = float(tid);
  __syncthreads();

  float acc = 0.f;
  for (int it = 0; it < REPEAT; ++it) {
    acc += s[idx];
    s[idx] = acc;
  }
  out[gtid] = acc;
}

// 用 padding 消除冲突：对 stride=16 的布局加 “+1” 偏移（打乱 bank 对齐）
// 思想：把线性地址从 idx*16 改成 idx*(16+1)，使得 (bank = index % 32) 更均匀分布
__global__ void kernel_padded_stride16(float* out) {
  // 注意：共享内存需求稍增，因为我们给每个逻辑“列”加了 1 的填充
  __shared__ volatile float s[BLOCK_SIZE * (16 + 1)];
  int tid = threadIdx.x;
  int gtid = blockIdx.x * blockDim.x + tid;

  int idx = tid * (16 + 1);
  s[idx] = float(tid);
  __syncthreads();

  float acc = 0.f;
  for (int it = 0; it < REPEAT; ++it) {
    acc += s[idx];
    s[idx] = acc;
  }
  out[gtid] = acc;
}

float run_and_time(void (*kernel)(float*), const char* name, float* d_out) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaDeviceSynchronize());

  CHECK(cudaEventRecord(start));
  kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_out);
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));

  float ms = 0.f;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaDeviceSynchronize());

  printf("%-28s : %8.3f ms\n", name, ms);
  return ms;
}

int main() {
  // 预热并打印设备信息
  int dev = 0;
  CHECK(cudaSetDevice(dev));
  cudaDeviceProp prop{};
  CHECK(cudaGetDeviceProperties(&prop, dev));
  printf("GPU: %s\nSM count: %d, SharedMem/Block: %zu KB\n\n", prop.name, prop.multiProcessorCount,
      prop.sharedMemPerBlockOptin / 1024);

  // 为输出分配显存（防止编译器把计算优化掉）
  float* d_out = nullptr;
  CHECK(cudaMalloc(&d_out, sizeof(float) * GRID_SIZE * BLOCK_SIZE));

  // 预热一次（减少首次启动抖动）
  kernel_no_conflict<<<GRID_SIZE, BLOCK_SIZE>>>(d_out);
  CHECK(cudaDeviceSynchronize());

  // 测试并计时
  run_and_time(kernel_no_conflict, "No conflict (stride=1)", d_out);
  run_and_time(kernel_conflict_stride2, "2-way conflict (stride=2)", d_out);
  run_and_time(kernel_conflict_stride16, "16-way conflict (stride=16)", d_out);
  run_and_time(kernel_padded_stride16, "Padded fix (stride=16,+1)", d_out);

  CHECK(cudaFree(d_out));
  return 0;
}
