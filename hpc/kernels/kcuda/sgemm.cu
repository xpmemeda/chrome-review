#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <cuda_runtime.h>

#include "./module.h"

namespace {

constexpr int BM = 128;       // Block tile M
constexpr int BN = 128;       // Block tile N
constexpr int BK = 32;        // Block tile K
constexpr int THREADS = 256;  // 256
constexpr int TM = 8;         // Thread Tile M
constexpr int TN = 8;         // Thread Tile N

__host__ __device__ inline int idx2d(int r, int c, int ld) { return r * ld + c; }

__global__ void sgemm_naive_kernel(
    const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  static_assert(BM % TM == 0 && BN % TN == 0);

  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int tx = threadIdx.x;
  const int num_works = (BM / TM) * (BN / TN);

  for (int kk = 0; kk < K; kk += BK) {
    // A: BM x BK
    for (int i = tx; i < BM * BK; i += THREADS) {
      int r = i / BK;
      int c = i % BK;
      int gr = block_row * BM + r;
      int gc = kk + c;
      As[r][c] = (gr < M && gc < K) ? A[idx2d(gr, gc, K)] : 0.0f;
    }
    // B: BK x BN
    for (int j = tx; j < BK * BN; j += THREADS) {
      int r = j / BN;
      int c = j % BN;
      int gr = kk + r;
      int gc = block_col * BN + c;
      Bs[r][c] = (gr < K && gc < N) ? B[idx2d(gr, gc, N)] : 0.0f;
    }
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

void sgemm_naive(at::Tensor r, at::Tensor x, at::Tensor w) {
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
  sgemm_naive_kernel<<<grid, block>>>(x.data_ptr<float>(), w.data_ptr<float>(), r.data_ptr<float>(), M, N, K);
}

static cr::Register _([](pybind11::module& m) { m.def("sgemm_naive", &sgemm_naive); });

}  // namespace
