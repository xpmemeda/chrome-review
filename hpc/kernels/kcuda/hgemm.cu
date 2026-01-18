#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/half.h"
#include "cutlass/layout/matrix.h"

#include "./module.h"

namespace {

__host__ __device__ inline int idx2d(int r, int c, int ld) { return r * ld + c; }

template <typename T, int M, int N, int NUM_THREADS, bool ZERO_OUT_OF_BOUND = true>
struct CopyNaive {
  __forceinline__ __device__ CopyNaive(
      const T* __restrict__ src, T* __restrict__ dst, int src_stride, int dst_stride, int m, int n)
      : src_(src), dst_(dst), src_stride_(src_stride), dst_stride_(dst_stride), m_(m), n_(n) {}

  __forceinline__ __device__ void operator()() {
    constexpr int NUM_WORKS = M * N;
    /*
    NOTE: 把 widx += blockIdx.x 替换为 widx += NUM_THREADS 可以获得巨大的性能提升。

    循环步长变成编译期常量 → 强化强度削减（strength reduction）
    原来每次迭代都要做：

    widx_r = widx / N;   // 除法
    widx_c = widx % N;   // 取模
    idx = widx_r*ld + widx_c; // 乘法+加法


    虽然 N 是模板常量，但当“步长”不固定（+= blockDim.x）时，编译器很难把 r/c/idx 写成固定增量的递推。
    现在步长是常量 NUM_THREADS，编译器可以把“除法/取模/乘法”在循环里消失为：

    每次循环让 (r,c) 用固定增量推进（带进位/回卷）；

    或者直接对 linear_index/地址指针做固定字节数的指针累加。
    结果是每轮只剩几条 add/mad.wide，整数 ALU 压力、寄存器压力都下降。

    更激进的循环展开 & 指令调度
    步长、总工作量（NUM_WORKS=M*N）都成了常量，ptxas
    能做部分或完全展开，把加载/写回与地址更新流水化，减少分支与回边开销。使用 blockDim.x 时，这些优化会被大幅克制。

    去掉读取特殊寄存器的依赖
    blockDim.x 实际来自特殊寄存器 %ntid.x，会多一条 mov.u32 以及随之而来的数据依赖。常量化后，这条依赖消失，调度更自由。

    （常见）消除“尾部判断”的分支
    若 NUM_WORKS % NUM_THREADS == 0，许多“最后一轮是否越界”的判断可以在编译期直接消掉，控制流更干净。
    （即使不整除，常量信息也能让编译器生成代价更低的掩码/分支。）
    */
    for (int widx = threadIdx.x; widx < NUM_WORKS; widx += NUM_THREADS) {
      int widx_r = widx / N;
      int widx_c = widx % N;
      if (widx_r < m_ && widx_c < n_) {
        dst_[idx2d(widx_r, widx_c, dst_stride_)] = src_[idx2d(widx_r, widx_c, src_stride_)];
      } else {
        if constexpr (ZERO_OUT_OF_BOUND) {
          dst_[idx2d(widx_r, widx_c, dst_stride_)] = T(0);
        }
      }
    }
  }

  const T* __restrict__ src_;
  T* __restrict__ dst_;
  int src_stride_;
  int dst_stride_;
  int m_;
  int n_;
};

template <typename Kernel>
class KernelProp {
 public:
  KernelProp(const char* name, Kernel kernel) {
    cudaFuncAttributes attr{};
    cudaError_t err = cudaFuncGetAttributes(&attr, (const void*)kernel);
    if (err == cudaSuccess) {
      printf("[%s] binaryVersion=sm_%d, ptxVersion=%d\n", name, attr.binaryVersion, attr.ptxVersion);
    } else {
      printf("cudaFuncGetAttributes failed: %s\n", cudaGetErrorString(err));
    }
  }
};

namespace naive {

constexpr int BM = 128;       // Block tile M
constexpr int BN = 128;       // Block tile N
constexpr int BK = 32;        // Block tile K
constexpr int THREADS = 256;  // 256
constexpr int TM = 8;         // Thread Tile M
constexpr int TN = 8;         // Thread Tile N

__global__ void hgemm_naive_kernel(
    const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  static_assert(BM % TM == 0 && BN % TN == 0);

  __shared__ half As[BM][BK];
  __shared__ half Bs[BK][BN];

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int tx = threadIdx.x;
  const int num_works = (BM / TM) * (BN / TN);

  for (int kk = 0; kk < K; kk += BK) {
    CopyNaive<half, BM, BK, THREADS> copy_a(A + block_row * BM * K + kk, &As[0][0], K, BK, M - block_row * BM, N - kk);
    copy_a();
    CopyNaive<half, BK, BN, THREADS> copy_b(B + kk * N + block_col * BN, &Bs[0][0], N, BN, K - kk, N - block_col * BN);
    copy_b();
    __syncthreads();

    for (int th_linear = tx; th_linear < num_works; th_linear += blockDim.x) {
      float acc[TM][TN] = {0};

      const int widx_m = (th_linear / (BN / TN)) * TM;
      const int widx_n = (th_linear % (BN / TN)) * TN;

      const int row0 = block_row * BM + widx_m;
      const int col0 = block_col * BN + widx_n;

      // acc += As * Bs
      for (int k_inner = 0; k_inner < BK; ++k_inner) {
        float a_frag[TM];
#pragma unroll
        for (int im = 0; im < TM; ++im) {
          int r = widx_m + im;
          a_frag[im] = As[r][k_inner];
        }
        float b_frag[TN];
#pragma unroll
        for (int jn = 0; jn < TN; ++jn) {
          int c = widx_n + jn;
          b_frag[jn] = Bs[k_inner][c];
        }
#pragma unroll
        for (int im = 0; im < TM; ++im)
#pragma unroll
          for (int jn = 0; jn < TN; ++jn) {
            acc[im][jn] += a_frag[im] * b_frag[jn];
          }
      }

      __syncthreads();

// C
#pragma unroll
      for (int im = 0; im < TM; ++im) {
        int r = row0 + im;
        if (r < M) {
#pragma unroll
          for (int jn = 0; jn < TN; ++jn) {
            int c = col0 + jn;
            if (c < N) {
              if (kk == 0) {
                C[idx2d(r, c, N)] = acc[im][jn];
              } else {
                C[idx2d(r, c, N)] += acc[im][jn];
              }
            }
          }
        }
      }
    }
  }
}

void hgemm_naive(at::Tensor r, at::Tensor x, at::Tensor w) {
  if (r.ndimension() > 2) {
    auto s = r.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int N = s.back();
    r = r.reshape({M, N});
  }
  if (x.ndimension() > 2) {
    auto s = x.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int K = s.back();
    x = x.reshape({M, K});
  }

  if (r.ndimension() != 2 || x.ndimension() != 2 || w.ndimension() != 2) {
    throw std::runtime_error("rank != 2");
  }
  if (r.size(0) != x.size(0)) {
    throw std::runtime_error("M not match.");
  }
  if (r.size(1) != w.size(1)) {
    throw std::runtime_error("N not match.");
  }
  if (x.size(1) != w.size(0)) {
    throw std::runtime_error("K not match.");
  }

  const unsigned M = r.size(r.ndimension() - 2);
  const unsigned N = r.size(r.ndimension() - 1);
  const unsigned K = x.size(x.ndimension() - 1);

  const unsigned gridx = (N + BN - 1) / BN;
  const unsigned gridy = (M + BM - 1) / BM;
  const unsigned gridz = 1;
  const unsigned blockx = THREADS;
  const unsigned blocky = 1;
  const unsigned blockz = 1;
  dim3 grid(gridx, gridy, gridz);
  dim3 block(blockx, blocky, blockz);
  hgemm_naive_kernel<<<grid, block>>>(
      reinterpret_cast<half*>(x.data_ptr()), reinterpret_cast<half*>(w.data_ptr()), r.data_ptr<float>(), M, N, K);
}

static cr::Register _([](pybind11::module& m) { m.def("hgemm_naive", &hgemm_naive); });

}  // namespace naive

namespace wmma {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;
constexpr int WARP_SIZE = 32;

constexpr int NUM_WARP_PER_BLOCK_M = BLOCK_M / WARP_M;
constexpr int NUM_WARP_PER_BLOCK_N = BLOCK_N / WARP_N;
constexpr int THREADS = NUM_WARP_PER_BLOCK_M * NUM_WARP_PER_BLOCK_N * WARP_SIZE;

__global__ void hgemm_wmma_kernel(half* A, half* B, float* C, int M, int N, int K, float alpha, float beta) {
  constexpr int NUM_WMMA_PER_WARP_M = WARP_M / WMMA_M;
  constexpr int NUM_WMMA_PER_WARP_N = WARP_N / WMMA_N;
  constexpr int NUM_WMMA_PER_BLOCK_K = BLOCK_K / WMMA_K;

  const int block_r = blockIdx.y;
  const int block_c = blockIdx.x;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int warp_r = warp_idx / NUM_WARP_PER_BLOCK_N;
  const int warp_c = warp_idx % NUM_WARP_PER_BLOCK_N;

  // NOTE: static too large sharedmemory will cause invalid arguments err.
  extern __shared__ half sm[];
  auto Cs = reinterpret_cast<float (*)[BLOCK_N]>(sm);
  auto As = reinterpret_cast<half(*)[BLOCK_K]>(sm);
  auto Bs = reinterpret_cast<half(*)[BLOCK_N]>(sm + BLOCK_M * BLOCK_K);

  auto Cg = C + block_r * BLOCK_M * N + block_c * BLOCK_N;
  auto blockM = min(BLOCK_M, M - block_r * BLOCK_M);
  auto blockN = min(BLOCK_N, N - block_c * BLOCK_N);

  // Load C.
  CopyNaive<float, BLOCK_M, BLOCK_N, THREADS> rc(Cg, reinterpret_cast<float*>(Cs), N, BLOCK_N, blockM, blockN);
  rc();
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_fragments[NUM_WMMA_PER_WARP_M]
                                                                                              [NUM_WMMA_PER_WARP_N];
#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
      auto ptrC = &Cs[warp_r * WARP_M + wmma_r * WMMA_M][warp_c * WARP_N + wmma_c * WMMA_N];
      nvcuda::wmma::load_matrix_sync(c_fragments[wmma_r][wmma_c], ptrC, BLOCK_N, nvcuda::wmma::mem_row_major);
#pragma unroll
      for (int t = 0; t < c_fragments[wmma_r][wmma_c].num_elements; ++t) {
        c_fragments[wmma_r][wmma_c].x[t] *= beta / alpha;
      }
    }
  }
  __syncthreads();

  for (int k = 0; k < K; k += BLOCK_K) {
    CopyNaive<half, BLOCK_M, BLOCK_K, THREADS> ra(
        A + block_r * BLOCK_M * K + k, &As[0][0], K, BLOCK_K, M - block_r * BLOCK_M, K - k);
    ra();
    CopyNaive<half, BLOCK_K, BLOCK_N, THREADS> rb(
        B + k * N + block_c * BLOCK_N, &Bs[0][0], N, BLOCK_N, K - k, N - block_c * BLOCK_N);
    rb();
    __syncthreads();

    for (int wmma_k = 0; wmma_k < NUM_WMMA_PER_BLOCK_K; ++wmma_k) {
      // Load A.
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
          a_fragments[NUM_WMMA_PER_WARP_M];
#pragma unroll
      for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
        nvcuda::wmma::load_matrix_sync(
            a_fragments[wmma_r], &As[warp_r * WARP_M + wmma_r * WMMA_M][wmma_k * WMMA_K], BLOCK_K);
      }
      // Load B.
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
          b_fragments[NUM_WMMA_PER_WARP_N];
#pragma unroll
      for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
        nvcuda::wmma::load_matrix_sync(
            b_fragments[wmma_c], &Bs[wmma_k * WMMA_K][warp_c * WARP_N + wmma_c * WMMA_N], BLOCK_N);
      }
#pragma unroll
      for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
        for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
          nvcuda::wmma::mma_sync(
              c_fragments[wmma_r][wmma_c], a_fragments[wmma_r], b_fragments[wmma_c], c_fragments[wmma_r][wmma_c]);
        }
      }
    }
  }

#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
#pragma unroll
      for (int t = 0; t < c_fragments[wmma_r][wmma_c].num_elements; ++t) {
        c_fragments[wmma_r][wmma_c].x[t] *= alpha;
      }
    }
  }

  __syncthreads();

  // Store C.
#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
      auto ptrC = &Cs[warp_r * WARP_M + wmma_r * WMMA_M][warp_c * WARP_N + wmma_c * WMMA_N];
      nvcuda::wmma::store_matrix_sync(ptrC, c_fragments[wmma_r][wmma_c], BLOCK_N, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  CopyNaive<float, BLOCK_M, BLOCK_N, THREADS, false> wc(reinterpret_cast<float*>(Cs), Cg, BLOCK_N, N, blockM, blockN);
  wc();
}

struct DynamicMemoryGuard {
  DynamicMemoryGuard() {
    cudaDeviceProp prop;
    cudaError_t r;
    r = cudaGetDeviceProperties(&prop, 0);
    if (r != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties err.");
    }
    constexpr int required_sharedmemory_size =
        std::max<int>(BLOCK_M * BLOCK_N * sizeof(float), (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half));
    printf(
        "prop.sharedMemPerMultiprocessor %zu prop.sharedMemPerBlock %zu required_sharedmemory_size "
        "%d\n",
        prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlock, required_sharedmemory_size);
    r = cudaFuncSetAttribute(
        hgemm_wmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, required_sharedmemory_size);
    if (r != cudaSuccess) {
      throw std::runtime_error("cudaFuncSetAttribute err.");
    }
  }
};

void hgemm_wmma(at::Tensor a, at::Tensor b, at::Tensor c, float alpha, float beta) {
  if (c.ndimension() > 2) {
    auto s = c.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int N = s.back();
    c = c.reshape({M, N});
  }
  if (a.ndimension() > 2) {
    auto s = a.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int K = s.back();
    a = a.reshape({M, K});
  }

  if (c.ndimension() != 2 || a.ndimension() != 2 || b.ndimension() != 2) {
    throw std::runtime_error("rank != 2");
  }
  if (c.size(0) != a.size(0)) {
    throw std::runtime_error("M not match.");
  }
  if (c.size(1) != b.size(1)) {
    throw std::runtime_error("N not match.");
  }
  if (a.size(1) != b.size(0)) {
    throw std::runtime_error("K not match.");
  }

  const unsigned M = c.size(c.ndimension() - 2);
  const unsigned N = c.size(c.ndimension() - 1);
  const unsigned K = a.size(a.ndimension() - 1);

  static DynamicMemoryGuard dynamic_memory_guard;

  auto cdiv = [](int x, int y) { return (x + y - 1) / y; };
  dim3 grid(cdiv(N, BLOCK_N), cdiv(M, BLOCK_M), 1);
  dim3 block(THREADS, 1, 1);
  constexpr int sharedmemory_size =
      std::max<int>(BLOCK_M * BLOCK_N * sizeof(float), (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half));

  hgemm_wmma_kernel<<<grid, block, sharedmemory_size>>>(reinterpret_cast<half*>(a.data_ptr()),
      reinterpret_cast<half*>(b.data_ptr()), c.data_ptr<float>(), M, N, K, alpha, beta);
}

static cr::Register _([](pybind11::module& m) { m.def("hgemm_wmma", &hgemm_wmma); });

}  // namespace wmma

namespace wmma_asyncp {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;
constexpr int WARP_SIZE = 32;

constexpr int GROUP_SIZE_N = BLOCK_N * 8;

constexpr int NUM_WARP_PER_BLOCK_M = BLOCK_M / WARP_M;
constexpr int NUM_WARP_PER_BLOCK_N = BLOCK_N / WARP_N;
constexpr int THREADS = NUM_WARP_PER_BLOCK_M * NUM_WARP_PER_BLOCK_N * WARP_SIZE;

static __device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
  // NOTE: n: immediate constant r: 32bit l: 64bit
  // NOTE: ca:G->L1->R cg:G->R
  // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
#else
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

static __device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__global__ void hgemm_wmma_cpasync_kernel(half* A, half* B, float* C, int M, int N, int K, float alpha, float beta) {
  constexpr int NUM_WMMA_PER_WARP_M = WARP_M / WMMA_M;
  constexpr int NUM_WMMA_PER_WARP_N = WARP_N / WMMA_N;
  constexpr int NUM_WMMA_PER_BLOCK_K = BLOCK_K / WMMA_K;

  const int num_blocks_m = (M + BLOCK_M - 1) / BLOCK_M;
  const int num_blocks_n = (N + BLOCK_N - 1) / BLOCK_N;

  // TODO: L2 optimize.
  // static_assert(GROUP_SIZE_N % BLOCK_N == 0);
  // const int num_blocks_per_group_m = num_blocks_m;
  // const int num_blocks_per_group_n = GROUP_SIZE_N / BLOCK_N;
  // const int num_blocks_per_group = num_blocks_per_group_m * num_blocks_per_group_n;
  // const int group_idx = blockIdx.x / num_blocks_per_group;
  // const int group_off = blockIdx.x % num_blocks_per_group;
  // const int block_r = group_off / num_blocks_per_group_n;
  // const int block_c = group_off % num_blocks_per_group_n + group_idx * num_blocks_per_group_n;

  const int block_r = blockIdx.x / num_blocks_n;
  const int block_c = blockIdx.x % num_blocks_n;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int warp_r = warp_idx / NUM_WARP_PER_BLOCK_N;
  const int warp_c = warp_idx % NUM_WARP_PER_BLOCK_N;

  constexpr int bytesAsPerStage = BLOCK_M * BLOCK_K * sizeof(half);
  constexpr int bytesBsPerStage = BLOCK_K * BLOCK_N * sizeof(half);
  static_assert(bytesAsPerStage % 16 == 0 && bytesBsPerStage % 16 == 0);
  constexpr int num_chunks_A = bytesAsPerStage / 16;
  constexpr int num_chunks_B = bytesBsPerStage / 16;

  // NOTE: static too large sharedmemory will cause invalid arguments err.
  extern __shared__ uint8_t sm[];
  auto Cs = reinterpret_cast<float (*)[BLOCK_N]>(sm);
  auto As0 = reinterpret_cast<half(*)[BLOCK_K]>(sm);
  auto As1 = reinterpret_cast<half(*)[BLOCK_K]>(sm + bytesAsPerStage);
  auto Bs0 = reinterpret_cast<half(*)[BLOCK_N]>(sm + 2 * bytesAsPerStage);
  auto Bs1 = reinterpret_cast<half(*)[BLOCK_N]>(sm + 2 * bytesAsPerStage + bytesBsPerStage);

  auto Cg = C + block_r * BLOCK_M * N + block_c * BLOCK_N;
  auto blockM = min(BLOCK_M, M - block_r * BLOCK_M);
  auto blockN = min(BLOCK_N, N - block_c * BLOCK_N);
  auto blockK = min(BLOCK_K, K);

  // Load C.
  CopyNaive<float, BLOCK_M, BLOCK_N, THREADS> rc(Cg, reinterpret_cast<float*>(Cs), N, BLOCK_N, blockM, blockN);
  rc();
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_fragments[NUM_WMMA_PER_WARP_M]
                                                                                              [NUM_WMMA_PER_WARP_N];
#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
      auto ptrC = &Cs[warp_r * WARP_M + wmma_r * WMMA_M][warp_c * WARP_N + wmma_c * WMMA_N];
      nvcuda::wmma::load_matrix_sync(c_fragments[wmma_r][wmma_c], ptrC, BLOCK_N, nvcuda::wmma::mem_row_major);
#pragma unroll
      for (int t = 0; t < c_fragments[wmma_r][wmma_c].num_elements; ++t) {
        c_fragments[wmma_r][wmma_c].x[t] *= beta / alpha;
      }
    }
  }
  __syncthreads();

  // Load A.
  for (int chunk = threadIdx.x; chunk < num_chunks_A; chunk += THREADS) {
    int idx = chunk * 16;
    int r = (idx / sizeof(half)) / BLOCK_K;
    int c = (idx / sizeof(half)) % BLOCK_K;
    auto As = reinterpret_cast<uint8_t*>(As0) + idx;
    auto Ag = A + (block_r * BLOCK_M + r) * K + c;
    if (r < blockM && c < blockK) {
      cp_async_16B(As, Ag);
    } else {
      *reinterpret_cast<int4*>(As) = make_int4(0, 0, 0, 0);
    }
  }
  // Load B.
  for (int chunk = threadIdx.x; chunk < num_chunks_B; chunk += THREADS) {
    int idx = chunk * 16;
    int r = (idx / sizeof(half)) / BLOCK_N;
    int c = (idx / sizeof(half)) % BLOCK_N;
    auto Bs = reinterpret_cast<uint8_t*>(Bs0) + idx;
    auto Bg = B + r * N + block_c * BLOCK_N + c;
    if (r < blockK && c < blockN) {
      cp_async_16B(Bs, Bg);
    } else {
      *reinterpret_cast<int4*>(Bs) = make_int4(0, 0, 0, 0);
    }
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
      a_fragments[NUM_WMMA_PER_WARP_M];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, nvcuda::wmma::row_major>
      b_fragments[NUM_WMMA_PER_WARP_N];

  int stage = 0;
  for (int k = 0; k < K; k += BLOCK_K) {
    // Load next A & B.
    {
      int k_next = k + BLOCK_K;
      if (k_next < K) {
        auto AsN = stage == 0 ? As1 : As0;
        auto BsN = stage == 0 ? Bs1 : Bs0;
        int blockK = min(BLOCK_K, K - k);
        for (int chunk = threadIdx.x; chunk < num_chunks_A; chunk += THREADS) {
          int idx = chunk * 16;
          int r = (idx / sizeof(half)) / BLOCK_K;
          int c = (idx / sizeof(half)) % BLOCK_K;
          auto As = reinterpret_cast<uint8_t*>(AsN) + idx;
          auto Ag = A + (block_r * BLOCK_M + r) * K + k_next + c;
          if (r < blockM && c < blockK) {
            cp_async_16B(As, Ag);
          } else {
            *reinterpret_cast<int4*>(As) = make_int4(0, 0, 0, 0);
          }
        }
        for (int chunk = threadIdx.x; chunk < num_chunks_B; chunk += THREADS) {
          int idx = chunk * 16;
          int r = (idx / sizeof(half)) / BLOCK_N;
          int c = (idx / sizeof(half)) % BLOCK_N;
          auto Bs = reinterpret_cast<uint8_t*>(BsN) + idx;
          auto Bg = B + (k_next + r) * N + block_c * BLOCK_N + c;
          if (r < blockK && c < blockN) {
            cp_async_16B(Bs, Bg);
          } else {
            *reinterpret_cast<int4*>(Bs) = make_int4(0, 0, 0, 0);
          }
        }
        cp_async_commit_group();
      }
    }

    auto As = stage == 0 ? As0 : As1;
    auto Bs = stage == 0 ? Bs0 : Bs1;

    for (int wmma_k = 0; wmma_k < NUM_WMMA_PER_BLOCK_K; ++wmma_k) {
      // Load A.

#pragma unroll
      for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
        nvcuda::wmma::load_matrix_sync(
            a_fragments[wmma_r], &As[warp_r * WARP_M + wmma_r * WMMA_M][wmma_k * WMMA_K], BLOCK_K);
      }
      // Load B.
#pragma unroll
      for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
        nvcuda::wmma::load_matrix_sync(
            b_fragments[wmma_c], &Bs[wmma_k * WMMA_K][warp_c * WARP_N + wmma_c * WMMA_N], BLOCK_N);
      }
#pragma unroll
      for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
        for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
          nvcuda::wmma::mma_sync(
              c_fragments[wmma_r][wmma_c], a_fragments[wmma_r], b_fragments[wmma_c], c_fragments[wmma_r][wmma_c]);
        }
      }
    }

    {
      int k_next = k + BLOCK_K;
      if (k_next < K) {
        cp_async_wait_group<0>();
        __syncthreads();
        stage = (stage + 1) % 2;
      }
    }
  }

#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
#pragma unroll
      for (int t = 0; t < c_fragments[wmma_r][wmma_c].num_elements; ++t) {
        c_fragments[wmma_r][wmma_c].x[t] *= alpha;
      }
    }
  }

  __syncthreads();

  // Store C.
#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
      auto ptrC = &Cs[warp_r * WARP_M + wmma_r * WMMA_M][warp_c * WARP_N + wmma_c * WMMA_N];
      nvcuda::wmma::store_matrix_sync(ptrC, c_fragments[wmma_r][wmma_c], BLOCK_N, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  CopyNaive<float, BLOCK_M, BLOCK_N, THREADS, false> wc(reinterpret_cast<float*>(Cs), Cg, BLOCK_N, N, blockM, blockN);
  wc();
}

struct DynamicMemoryGuard {
  DynamicMemoryGuard() {
    cudaDeviceProp prop;
    cudaError_t r;
    r = cudaGetDeviceProperties(&prop, 0);
    if (r != cudaSuccess) {
      throw std::runtime_error("cudaGetDeviceProperties err.");
    }
    constexpr int required_sharedmemory_size =
        std::max<int>(BLOCK_M * BLOCK_N * sizeof(float), 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half));
    printf(
        "prop.sharedMemPerMultiprocessor %zu prop.sharedMemPerBlock %zu required_sharedmemory_size "
        "%d\n",
        prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlock, required_sharedmemory_size);
    r = cudaFuncSetAttribute(
        hgemm_wmma_cpasync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, required_sharedmemory_size);
    if (r != cudaSuccess) {
      throw std::runtime_error("cudaFuncSetAttribute err.");
    }
  }
};

void hgemm_wmma_cpasync(at::Tensor a, at::Tensor b, at::Tensor c, float alpha, float beta) {
  if (c.ndimension() > 2) {
    auto s = c.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int N = s.back();
    c = c.reshape({M, N});
  }
  if (a.ndimension() > 2) {
    auto s = a.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int K = s.back();
    a = a.reshape({M, K});
  }

  if (c.ndimension() != 2 || a.ndimension() != 2 || b.ndimension() != 2) {
    throw std::runtime_error("rank != 2");
  }
  if (c.size(0) != a.size(0)) {
    throw std::runtime_error("M not match.");
  }
  if (c.size(1) != b.size(1)) {
    throw std::runtime_error("N not match.");
  }
  if (a.size(1) != b.size(0)) {
    throw std::runtime_error("K not match.");
  }

  const unsigned M = c.size(c.ndimension() - 2);
  const unsigned N = c.size(c.ndimension() - 1);
  const unsigned K = a.size(a.ndimension() - 1);

  static DynamicMemoryGuard dynamic_memory_guard;
  static KernelProp kernel_prop("hgemm_wmma_cpasync_kernel", hgemm_wmma_cpasync_kernel);

  auto cdiv = [](int x, int y) { return (x + y - 1) / y; };
  dim3 grid(cdiv(N, BLOCK_N) * cdiv(M, BLOCK_M), 1, 1);
  dim3 block(THREADS, 1, 1);
  constexpr int sharedmemory_size =
      std::max<int>(BLOCK_M * BLOCK_N * sizeof(float), 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half));

  hgemm_wmma_cpasync_kernel<<<grid, block, sharedmemory_size>>>(reinterpret_cast<half*>(a.data_ptr()),
      reinterpret_cast<half*>(b.data_ptr()), c.data_ptr<float>(), M, N, K, alpha, beta);
}

static cr::Register _([](pybind11::module& m) { m.def("hgemm_wmma_cpasync", &hgemm_wmma_cpasync); });

}  // namespace wmma_asyncp

namespace mma_asyncp {

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 16;
constexpr int WARP_M = 64;
constexpr int WARP_N = 64;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 32;
constexpr int WARP_SIZE = 32;

constexpr int GROUP_SIZE_N = BLOCK_N * 8;

constexpr int NUM_WARP_PER_BLOCK_M = BLOCK_M / WARP_M;
constexpr int NUM_WARP_PER_BLOCK_N = BLOCK_N / WARP_N;
constexpr int THREADS = NUM_WARP_PER_BLOCK_M * NUM_WARP_PER_BLOCK_N * WARP_SIZE;

using Afragment = uint32_t[4];
using Bfragment = uint32_t[2];
using Cfragment = float[4];

__device__ __forceinline__ void dbgprint(int block, int thread, const char* prefix, Afragment& a) {
  if (blockIdx.x == block && threadIdx.x == thread) {
    float2 h01 = __half22float2(*reinterpret_cast<half2*>(&a[0]));
    float2 h23 = __half22float2(*reinterpret_cast<half2*>(&a[1]));
    float2 h45 = __half22float2(*reinterpret_cast<half2*>(&a[2]));
    float2 h67 = __half22float2(*reinterpret_cast<half2*>(&a[3]));
    printf("[%u %u] %s: %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n", blockIdx.x, threadIdx.x, prefix, h01.x, h01.y,
        h23.x, h23.y, h45.x, h45.y, h67.x, h67.y);
  }
}

__device__ __forceinline__ void dbgprint(int block, int thread, const char* prefix, Bfragment& b) {
  if (blockIdx.x == block && threadIdx.x == thread) {
    float2 h01 = __half22float2(*reinterpret_cast<half2*>(&b[0]));
    float2 h23 = __half22float2(*reinterpret_cast<half2*>(&b[1]));

    printf("[%u %u] %s: %.2f %.2f %.2f %.2f\n", blockIdx.x, threadIdx.x, prefix, h01.x, h01.y, h23.x, h23.y);
  }
}

__device__ __forceinline__ void dbgprint(int block, int thread, const char* prefix, Cfragment& c) {
  if (blockIdx.x == block && threadIdx.x == thread) {
    printf("[%u, %u] %s: %.2f %.2f %.2f %.2f\n", blockIdx.x, threadIdx.x, prefix, c[0], c[1], c[2], c[3]);
  }
}

static __device__ __forceinline__ void cp_async_16B(void* smem_ptr, const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
  uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
  // NOTE: n: immediate constant r: 32bit l: 64bit
  // NOTE: ca:G->L1->R cg:G->R
  // asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr), "l"(gmem_ptr));
#else
  *reinterpret_cast<int4*>(smem_ptr) = *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

static __device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n");
#endif
}

template <int N>
static __device__ __forceinline__ void cp_async_wait_group() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

__forceinline__ __device__ void ld_a_fragment(Afragment a_fragment, const half* wmma_addr, int ldm) {
  const int lane = threadIdx.x & 31;
  const half* addr = wmma_addr + (lane % 16) * ldm + (lane / 16) * 8;
  auto shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));

  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a_fragment[0]), "=r"(a_fragment[1]), "=r"(a_fragment[2]), "=r"(a_fragment[3])
      : "r"(shared_addr));
}

__forceinline__ __device__ void ld_b_fragment(Bfragment b_fragment, const half* wmma_addr, int ldm) {
  const int lane = threadIdx.x & 31;
  const half* addr = wmma_addr + (lane % 16) * ldm;
  uint32_t shared_addr = static_cast<uint32_t>(__cvta_generic_to_shared(addr));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
      : "=r"(b_fragment[0]), "=r"(b_fragment[1])
      : "r"(shared_addr));
}

__forceinline__ __device__ void warp_mma_m16n8k16_f16f16f32(
    Afragment a_fragment, Bfragment b_fragment, Cfragment c_fragment) {
#define HMMA16816_F32F16F16F32(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1) \
  asm volatile(                                                        \
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "             \
      "{%0,%1,%2,%3}, "                                                \
      "{%4,%5,%6,%7}, "                                                \
      "{%8,%9}, "                                                      \
      "{%0,%1,%2,%3};\n"                                               \
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)                         \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))

  HMMA16816_F32F16F16F32(c_fragment[0], c_fragment[1], c_fragment[2], c_fragment[3], a_fragment[0], a_fragment[1],
      a_fragment[2], a_fragment[3], b_fragment[0], b_fragment[1]);

#undef HMMA16816_F32F16F16F32
}

__device__ inline void warp_load_m16n8(float* sc_wmma_addr, int ldm, Cfragment c_fragment) {
  const int lane = threadIdx.x & 31;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync#mma-16816-c

  int r0 = (lane / 4);
  int r1 = r0 + 8;
  int c_pair = (lane % 4) * 2;
  int c0 = c_pair + 0;
  int c1 = c_pair + 1;

  c_fragment[0] = *(sc_wmma_addr + r0 * ldm + c0);
  c_fragment[1] = *(sc_wmma_addr + r0 * ldm + c1);
  c_fragment[2] = *(sc_wmma_addr + r1 * ldm + c0);
  c_fragment[3] = *(sc_wmma_addr + r1 * ldm + c1);
}

__device__ inline void warp_store_m16n8(float* sc_wmma_addr, int ldm, Cfragment c_fragment) {
  const int lane = threadIdx.x & 31;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=mma%2520sync#mma-16816-c

  int r0 = (lane / 4);
  int r1 = r0 + 8;
  int c_pair = (lane % 4) * 2;
  int c0 = c_pair + 0;
  int c1 = c_pair + 1;

  *(sc_wmma_addr + r0 * ldm + c0) = c_fragment[0];
  *(sc_wmma_addr + r0 * ldm + c1) = c_fragment[1];
  *(sc_wmma_addr + r1 * ldm + c0) = c_fragment[2];
  *(sc_wmma_addr + r1 * ldm + c1) = c_fragment[3];
}

__global__ void hgemm_mma_cpasync_kernel(half* A, half* B, float* C, int M, int N, int K, float alpha, float beta) {
  constexpr int NUM_WMMA_PER_WARP_M = WARP_M / WMMA_M;
  constexpr int NUM_WMMA_PER_WARP_N = WARP_N / WMMA_N;
  constexpr int NUM_WMMA_PER_BLOCK_K = BLOCK_K / WMMA_K;

  const int num_blocks_m = (M + BLOCK_M - 1) / BLOCK_M;
  const int num_blocks_n = (N + BLOCK_N - 1) / BLOCK_N;

  const int block_r = blockIdx.x / num_blocks_n;
  const int block_c = blockIdx.x % num_blocks_n;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int warp_r = warp_idx / NUM_WARP_PER_BLOCK_N;
  const int warp_c = warp_idx % NUM_WARP_PER_BLOCK_N;

  constexpr int bytesAsPerStage = BLOCK_M * BLOCK_K * sizeof(half);
  constexpr int bytesBsPerStage = BLOCK_K * BLOCK_N * sizeof(half);
  static_assert(bytesAsPerStage % 16 == 0 && bytesBsPerStage % 16 == 0);
  constexpr int num_chunks_A = bytesAsPerStage / 16;
  constexpr int num_chunks_B = bytesBsPerStage / 16;

  // NOTE: static too large sharedmemory will cause invalid arguments err.
  extern __shared__ uint8_t sm[];
  auto Cs = reinterpret_cast<float (*)[BLOCK_N]>(sm);
  auto As0 = reinterpret_cast<half(*)[BLOCK_K]>(sm);
  auto As1 = reinterpret_cast<half(*)[BLOCK_K]>(sm + bytesAsPerStage);
  auto Bs0 = reinterpret_cast<half(*)[BLOCK_N]>(sm + 2 * bytesAsPerStage);
  auto Bs1 = reinterpret_cast<half(*)[BLOCK_N]>(sm + 2 * bytesAsPerStage + bytesBsPerStage);

  auto Cg = C + block_r * BLOCK_M * N + block_c * BLOCK_N;
  auto blockM = min(BLOCK_M, M - block_r * BLOCK_M);
  auto blockN = min(BLOCK_N, N - block_c * BLOCK_N);
  auto blockK = min(BLOCK_K, K);

  Afragment a_fragments[NUM_WMMA_PER_WARP_M];
  Bfragment b_fragments[NUM_WMMA_PER_WARP_N];
  Cfragment c_fragments[NUM_WMMA_PER_WARP_M][NUM_WMMA_PER_WARP_N];

  // Load C, G->S
  CopyNaive<float, BLOCK_M, BLOCK_N, THREADS> rc(Cg, reinterpret_cast<float*>(Cs), N, BLOCK_N, blockM, blockN);
  rc();
  __syncthreads();
  // Load C, S->R
#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
      auto sc_wmma_addr = &Cs[warp_r * WARP_M + wmma_r * WMMA_M][warp_c * WARP_N + wmma_c * WMMA_N];
      Cfragment& cf = c_fragments[wmma_r][wmma_c];
      warp_load_m16n8(sc_wmma_addr, BLOCK_N, cf);
      cf[0] *= beta / alpha;
      cf[1] *= beta / alpha;
      cf[2] *= beta / alpha;
      cf[3] *= beta / alpha;
    }
  }
  __syncthreads();

  // Load A.
  for (int chunk = threadIdx.x; chunk < num_chunks_A; chunk += THREADS) {
    int idx = chunk * 16;
    int r = (idx / sizeof(half)) / BLOCK_K;
    int c = (idx / sizeof(half)) % BLOCK_K;
    auto As = reinterpret_cast<uint8_t*>(As0) + idx;
    auto Ag = A + (block_r * BLOCK_M + r) * K + c;
    if (r < blockM && c < blockK) {
      cp_async_16B(As, Ag);
    } else {
      *reinterpret_cast<int4*>(As) = make_int4(0, 0, 0, 0);
    }
  }
  // Load B.
  for (int chunk = threadIdx.x; chunk < num_chunks_B; chunk += THREADS) {
    int idx = chunk * 16;
    int r = (idx / sizeof(half)) / BLOCK_N;
    int c = (idx / sizeof(half)) % BLOCK_N;
    auto Bs = reinterpret_cast<uint8_t*>(Bs0) + idx;
    auto Bg = B + r * N + block_c * BLOCK_N + c;
    if (r < blockK && c < blockN) {
      cp_async_16B(Bs, Bg);
    } else {
      *reinterpret_cast<int4*>(Bs) = make_int4(0, 0, 0, 0);
    }
  }
  cp_async_commit_group();
  cp_async_wait_group<0>();
  __syncthreads();

  int stage = 0;
  for (int k = 0; k < K; k += BLOCK_K) {
    // Load next A & B.
    {
      int k_next = k + BLOCK_K;
      if (k_next < K) {
        auto AsN = stage == 0 ? As1 : As0;
        auto BsN = stage == 0 ? Bs1 : Bs0;
        int blockK = min(BLOCK_K, K - k);
        for (int chunk = threadIdx.x; chunk < num_chunks_A; chunk += THREADS) {
          int idx = chunk * 16;
          int r = (idx / sizeof(half)) / BLOCK_K;
          int c = (idx / sizeof(half)) % BLOCK_K;
          auto As = reinterpret_cast<uint8_t*>(AsN) + idx;
          auto Ag = A + (block_r * BLOCK_M + r) * K + k_next + c;
          if (r < blockM && c < blockK) {
            cp_async_16B(As, Ag);
          } else {
            *reinterpret_cast<int4*>(As) = make_int4(0, 0, 0, 0);
          }
        }
        for (int chunk = threadIdx.x; chunk < num_chunks_B; chunk += THREADS) {
          int idx = chunk * 16;
          int r = (idx / sizeof(half)) / BLOCK_N;
          int c = (idx / sizeof(half)) % BLOCK_N;
          auto Bs = reinterpret_cast<uint8_t*>(BsN) + idx;
          auto Bg = B + (k_next + r) * N + block_c * BLOCK_N + c;
          if (r < blockK && c < blockN) {
            cp_async_16B(Bs, Bg);
          } else {
            *reinterpret_cast<int4*>(Bs) = make_int4(0, 0, 0, 0);
          }
        }
        cp_async_commit_group();
      }
    }

    auto As = stage == 0 ? As0 : As1;
    auto Bs = stage == 0 ? Bs0 : Bs1;

    for (int wmma_k = 0; wmma_k < NUM_WMMA_PER_BLOCK_K; ++wmma_k) {
      // Load A.

#pragma unroll
      for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
        ld_a_fragment(a_fragments[wmma_r], &As[warp_r * WARP_M + wmma_r * WMMA_M][wmma_k * WMMA_K], BLOCK_K);
      }
      // Load B.
#pragma unroll
      for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
        ld_b_fragment(b_fragments[wmma_c], &Bs[wmma_k * WMMA_K][warp_c * WARP_N + wmma_c * WMMA_N], BLOCK_N);
      }

      // dbgprint(0, threadIdx.x & 31, "a", a_fragments[0]);
      // dbgprint(0, threadIdx.x & 31, "b", b_fragments[0]);

#pragma unroll
      for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
        for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
          warp_mma_m16n8k16_f16f16f32(a_fragments[wmma_r], b_fragments[wmma_c], c_fragments[wmma_r][wmma_c]);
        }
      }
    }

    // dbgprint(0, threadIdx.x & 31, "c", c_fragments[0][0]);

    {
      int k_next = k + BLOCK_K;
      if (k_next < K) {
        cp_async_wait_group<0>();
        __syncthreads();
        stage = (stage + 1) % 2;
      }
    }
  }

  __syncthreads();

  // Store C.
#pragma unroll
  for (int wmma_r = 0; wmma_r < NUM_WMMA_PER_WARP_M; ++wmma_r) {
#pragma unroll
    for (int wmma_c = 0; wmma_c < NUM_WMMA_PER_WARP_N; ++wmma_c) {
      auto sc_wmma = &Cs[warp_r * WARP_M + wmma_r * WMMA_M][warp_c * WARP_N + wmma_c * WMMA_N];
      Cfragment& cf = c_fragments[wmma_r][wmma_c];
      cf[0] *= alpha;
      cf[1] *= alpha;
      cf[2] *= alpha;
      cf[3] *= alpha;
      warp_store_m16n8(sc_wmma, BLOCK_N, cf);
    }
  }
  __syncthreads();
  CopyNaive<float, BLOCK_M, BLOCK_N, THREADS, false> wc(reinterpret_cast<float*>(Cs), Cg, BLOCK_N, N, blockM, blockN);
  wc();
}

void hgemm_mma_cpasync(at::Tensor a, at::Tensor b, at::Tensor c, float alpha, float beta) {
  if (c.ndimension() > 2) {
    auto s = c.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int N = s.back();
    c = c.reshape({M, N});
  }
  if (a.ndimension() > 2) {
    auto s = a.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int K = s.back();
    a = a.reshape({M, K});
  }

  if (c.ndimension() != 2 || a.ndimension() != 2 || b.ndimension() != 2) {
    throw std::runtime_error("rank != 2");
  }
  if (c.size(0) != a.size(0)) {
    throw std::runtime_error("M not match.");
  }
  if (c.size(1) != b.size(1)) {
    throw std::runtime_error("N not match.");
  }
  if (a.size(1) != b.size(0)) {
    throw std::runtime_error("K not match.");
  }

  const unsigned M = c.size(c.ndimension() - 2);
  const unsigned N = c.size(c.ndimension() - 1);
  const unsigned K = a.size(a.ndimension() - 1);

  struct DynamicMemoryGuard {
    DynamicMemoryGuard() {
      cudaDeviceProp prop;
      cudaError_t r;
      r = cudaGetDeviceProperties(&prop, 0);
      if (r != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties err.");
      }
      constexpr int required_sharedmemory_size =
          std::max<int>(BLOCK_M * BLOCK_N * sizeof(float), 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half));
      printf(
          "prop.sharedMemPerMultiprocessor %zu prop.sharedMemPerBlock %zu required_sharedmemory_size "
          "%d\n",
          prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlock, required_sharedmemory_size);
      r = cudaFuncSetAttribute(
          hgemm_mma_cpasync_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, required_sharedmemory_size);
      if (r != cudaSuccess) {
        throw std::runtime_error("cudaFuncSetAttribute err.");
      }
    }
  };

  static DynamicMemoryGuard dynamic_memory_guard;
  static KernelProp kernel_prop("hgemm_mma_cpasync_kernel", hgemm_mma_cpasync_kernel);

  auto cdiv = [](int x, int y) { return (x + y - 1) / y; };
  dim3 grid(cdiv(N, BLOCK_N) * cdiv(M, BLOCK_M), 1, 1);
  dim3 block(THREADS, 1, 1);
  constexpr int sharedmemory_size =
      std::max<int>(BLOCK_M * BLOCK_N * sizeof(float), 2 * (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(half));

  hgemm_mma_cpasync_kernel<<<grid, block, sharedmemory_size>>>(reinterpret_cast<half*>(a.data_ptr()),
      reinterpret_cast<half*>(b.data_ptr()), c.data_ptr<float>(), M, N, K, alpha, beta);
}

static cr::Register _([](pybind11::module& m) { m.def("hgemm_mma_cpasync", &hgemm_mma_cpasync); });

}  // namespace mma_asyncp

namespace cutlass_mma {

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = float;
using ElementAcc = float;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

static int const kElementsPerAccess = 4;
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<ElementC, kElementsPerAccess, ElementAcc, ElementC>;

using GemmOp = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAcc,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

cudaError_t hgemm_cutlass_kernel(const ElementA* A, const ElementB* B, ElementC* C, int M, int N, int K, ElementC alpha,
    ElementC beta, cudaStream_t stream = nullptr) {
  typename GemmOp::Arguments args({M, N, K},  // GemmProblemSize(M, N, K)
      {A, K},                                 // TensorRefA(ptr, lda)
      {B, N},                                 // TensorRefB(ptr, ldb)
      {C, N},                                 // TensorRefC(ptr, ldc)
      {C, N},                                 // TensorRefD(ptr, ldd)
      {alpha, beta}                           // epilogue params
  );

  GemmOp op;
  cutlass::Status status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "GemmOp cannot implement args: %d\n", int(status));
    return cudaErrorNotSupported;
  }

  status = op.initialize(args, stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "GemmOp initialize failed: %d\n", int(status));
    return cudaErrorUnknown;
  }

  status = op(stream);
  if (status != cutlass::Status::kSuccess) {
    fprintf(stderr, "GemmOp run failed: %d\n", int(status));
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

void hgemm_cutlass(at::Tensor a, at::Tensor b, at::Tensor c, float alpha, float beta) {
  if (c.ndimension() > 2) {
    auto s = c.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int N = s.back();
    c = c.reshape({M, N});
  }
  if (a.ndimension() > 2) {
    auto s = a.sizes();
    const int M = std::accumulate(s.begin(), s.end() - 1, 1, std::multiplies<unsigned>());
    const int K = s.back();
    a = a.reshape({M, K});
  }

  if (c.ndimension() != 2 || a.ndimension() != 2 || b.ndimension() != 2) {
    throw std::runtime_error("rank != 2");
  }
  if (c.size(0) != a.size(0)) {
    throw std::runtime_error("M not match.");
  }
  if (c.size(1) != b.size(1)) {
    throw std::runtime_error("N not match.");
  }
  if (a.size(1) != b.size(0)) {
    throw std::runtime_error("K not match.");
  }

  const unsigned M = c.size(c.ndimension() - 2);
  const unsigned N = c.size(c.ndimension() - 1);
  const unsigned K = a.size(a.ndimension() - 1);

  auto r = hgemm_cutlass_kernel(reinterpret_cast<ElementA*>(a.data_ptr()), reinterpret_cast<ElementB*>(b.data_ptr()),
      reinterpret_cast<ElementC*>(c.data_ptr()), M, N, K, alpha, beta);
  if (r != cudaSuccess) {
    throw std::runtime_error("hgemm cutlass err.");
  }
}

static cr::Register _([](pybind11::module& m) { m.def("hgemm_cutlass", &hgemm_cutlass); });

}  // namespace cutlass_mma

}  // namespace
