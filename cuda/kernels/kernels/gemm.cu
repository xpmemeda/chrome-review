#include "./gemm.h"

#include "ATen/cuda/CUDAContext.h"
#include "c10/cuda/CUDAStream.h"
#include "cublas_v2.h"
#include "mma.h"

#include "../support/assert.h"
#include "../support/exceptions.h"

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

using namespace nvcuda;
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define M WMMA_M
#define N WMMA_N
#define K WMMA_K
#define BLOCK_ROW_WARPS 2
#define BLOCK_COL_WARPS 4
#define WARP_ROW_TILES 4
#define WARP_COL_TILES 2
#define WARP_ROW_SIZE (WARP_ROW_TILES * M)
#define WARP_COL_SIZE (WARP_COL_TILES * N)
#define BLOCK_ROW_TILES (BLOCK_ROW_WARPS * WARP_ROW_TILES)  // 8
#define BLOCK_COL_TILES (BLOCK_COL_WARPS * WARP_COL_TILES)  // 8
#define BLOCK_ROW_SIZE (BLOCK_ROW_TILES * M)                // 128
#define BLOCK_COL_SIZE (BLOCK_COL_TILES * N)                // 128
#define NUM_WARPS (BLOCK_ROW_WARPS * BLOCK_COL_WARPS)       // 8
#define WARP_SIZE 32
#define NUM_THREADS (NUM_WARPS * WARP_SIZE)  // 256
#define SKEW_HALF 16
#define MAX(a, b) (a > b ? a : b)
#define CEIL_DIV(a, b) ((a + b - 1) / b)
// SM <= 64K ? CHUNK_K = 4 : CHUNK_K = 8
// ABC -> (?x?)BLOCKs(128x128) -> (2x4)WARPs(64x32)(GPU Warps Working Unit) -> (4x2)TILEs(16x16)
template <int CHUNK_K>
__global__ void gemm_kernel_v3(
    half* a, half* b, float* c, unsigned m, unsigned n, unsigned k, float alpha, float beta) {
  // Row major: A(m, k) B(k, n) -> C(m, n)
  // Col major: B(n, k) A(k, m) -> C(n, m)
  // The leading dimension always refers to the length of the first dimension of the array
  // A & B
  // extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];  // FIXME. Bug.
  extern __shared__ half shmem[];
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane_id = threadIdx.x % WARP_SIZE;
  // Exchange C or D between shared memory and global memory.
  float* shmem_warp_stream_ptr = reinterpret_cast<float*>(shmem) + warp_id * BLOCK_COL_SIZE * M;
  // Exchange C or D between shared memory and registers.
  const int warp_i = warp_id / BLOCK_COL_WARPS;
  const int warp_j = warp_id % BLOCK_COL_WARPS;
  float* shmem_warp_tile_ptr =
      reinterpret_cast<float*>(shmem) + warp_i * WARP_ROW_SIZE * BLOCK_COL_SIZE + warp_j * WARP_COL_SIZE;

  const unsigned block_num_m = CEIL_DIV(m, BLOCK_ROW_SIZE);
  const unsigned block_num_n = CEIL_DIV(n, BLOCK_COL_SIZE);
  for (unsigned block_id = blockIdx.x;; block_id += gridDim.x) {
    const unsigned block_i = block_id / block_num_n;
    const unsigned block_j = block_id % block_num_n;
    if (block_i >= block_num_m) {
      break;
    }
    // Copy C from global memory to shared memory. Each warp copys one line of tiles.
    const unsigned gemm_idx = (block_i * BLOCK_ROW_SIZE + warp_id * M) * n + block_j * BLOCK_COL_SIZE;
    float* src_gmem_warp_stream_ptr = &c[gemm_idx];
#pragma unroll
    for (int i = 0; i < M; ++i) {
      // C type is float.
      // 128 * 4bytes / 32 = 16bytes = 4 * 4(bytes) -> int4. Each thread copies exactly 4 elements.
      using copy_t = int4;
      auto smem_i_ptr = reinterpret_cast<copy_t*>(shmem_warp_stream_ptr + BLOCK_COL_SIZE * i);
      auto gmem_i_ptr = reinterpret_cast<copy_t*>(src_gmem_warp_stream_ptr + n * i);
      smem_i_ptr[lane_id] = gmem_i_ptr[lane_id];
    }
    __syncthreads();

    // One tile maps to one fragment.
    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c_fragments[WARP_ROW_TILES][WARP_COL_TILES];
    // Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_COL_TILES; ++j) {
        const float* tile_ptr = shmem_warp_tile_ptr + i * M * BLOCK_COL_SIZE + j * N;
        wmma::load_matrix_sync(c_fragments[i][j], tile_ptr, BLOCK_COL_SIZE, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Scale c fragments
    beta /= alpha;
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_COL_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c_fragments[i][j].num_elements; t++) {
          c_fragments[i][j].x[t] *= beta;
        }
      }
    }

    // CHUNK: Unit of A and B along k dimension.
    constexpr int CHUNK_SIZE = CHUNK_K * K;                   // global memory.
    constexpr int CHUNK_SKEW_SIZE = CHUNK_K * K + SKEW_HALF;  // shared memory.
    const int num_chunks = CEIL_DIV(k, CHUNK_SIZE);
    for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      // Load A & B matrix.
      if (warp_id < 4) {
        // A: BLOCK_ROW_SIZE x CHUNK_SIZE
        // Warps 0-3 copy the A matrix, each warp copys 2 line of tiles. A: 16x8TILEs(MxK) C: 16x16TILEs(MxN)
        const half* warp_base_ptr = a + block_i * BLOCK_ROW_SIZE * k + (warp_id % 4) * M * k * 2;
        const half* lane_group_base_ptr = warp_base_ptr + (lane_id / 16) * M * k;
        size_t shmem_lane_group_idx = ((warp_id % 4) * M * 2 + lane_id / 16) * CHUNK_SKEW_SIZE;
        const half* lane_group_ptr = lane_group_base_ptr + chunk_idx * CHUNK_SIZE;
#pragma unroll
        for (int i = 0; i < M; ++i) {
          using copy_t = int4;
          auto smem_i_ptr = reinterpret_cast<copy_t*>(shmem + shmem_lane_group_idx + i * CHUNK_SKEW_SIZE);
          auto gmem_i_ptr = reinterpret_cast<const copy_t*>(lane_group_ptr + i * k);
          smem_i_ptr[lane_id % 16] = gmem_i_ptr[lane_id % 16];
        }
      } else {
        // B: CHUNK_SIZE x BLOCK_COL_SIZE
        // warps 4-7 copy the B matrix. each warp copys 1 line of tiles. B: 8x16TILEs(KxN) C: 16x16TILEs(MxN)
        const half* warp_base_ptr = b + block_j * BLOCK_COL_SIZE + (warp_id % 4) * K * n;
        // smem ptr, B offset is the area of CHUNK A.
        size_t shmem_b_off = BLOCK_ROW_SIZE * CHUNK_SKEW_SIZE;
        size_t shmem_warp_idx = shmem_b_off + (warp_id % 4) * K * BLOCK_COL_SIZE;
        // gmem ptr.
        const half* warp_ptr = warp_base_ptr + chunk_idx * CHUNK_SIZE * n;
#pragma unroll
        for (int i = 0; i < K; ++i) {
          using copy_t = int4;
          auto smem_i_ptr = reinterpret_cast<copy_t*>(shmem + shmem_warp_idx + i * BLOCK_COL_SIZE);
          auto gmem_i_ptr = reinterpret_cast<const copy_t*>(warp_ptr + i * n);
          smem_i_ptr[lane_id] = gmem_i_ptr[lane_id];
        }
      }

      __syncthreads();

      // Compute C.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; ++k_step) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_fragments[WARP_ROW_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_fragments[WARP_COL_TILES];
#pragma unroll
        // Load A from shared memory to registers.
        for (int i = 0; i < WARP_ROW_TILES; ++i) {
          size_t tile_a_i = (warp_i * WARP_ROW_TILES + i) * M * CHUNK_SKEW_SIZE + k_step * K;
          wmma::load_matrix_sync(a_fragments[i], reinterpret_cast<half*>(shmem) + tile_a_i, CHUNK_SKEW_SIZE);
        }
#pragma unroll
        // Load B from shared memory to registers.
        for (int j = 0; j < WARP_COL_TILES; ++j) {
          size_t shmem_b_off = BLOCK_ROW_SIZE * CHUNK_SKEW_SIZE;
          size_t tile_b_j = shmem_b_off + k_step * BLOCK_COL_SIZE + (warp_j * WARP_COL_TILES + j) * N;
          wmma::load_matrix_sync(b_fragments[j], reinterpret_cast<half*>(shmem) + tile_b_j, BLOCK_COL_SIZE);
        }
#pragma unroll
        for (int i = 0; i < WARP_ROW_TILES; ++i) {
          for (int j = 0; j < WARP_COL_TILES; ++j) {
            wmma::mma_sync(c_fragments[i][j], a_fragments[i], b_fragments[j], c_fragments[i][j]);
          }
        }
      }
      __syncthreads();
    }

    // Store D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_ROW_TILES; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_COL_TILES; ++j) {
#pragma unroll
        for (int t = 0; t < c_fragments[i][j].num_elements; ++t) {
          c_fragments[i][j].x[t] *= alpha;
        }
        float* tile_ptr = shmem_warp_tile_ptr + i * M * BLOCK_COL_SIZE + j * N;
        wmma::store_matrix_sync(tile_ptr, c_fragments[i][j], BLOCK_COL_SIZE, wmma::mem_row_major);
      }
    }
    __syncthreads();

    // Stream D to global memory.
#pragma unroll
    for (int i = 0; i < M; ++i) {
      // C type is float.
      // 128 * 4bytes / 32 = 16bytes = 4 * 4(bytes) -> int4. Each thread copies exactly 4 elements.
      using copy_t = int4;
      auto smem_i_ptr = reinterpret_cast<copy_t*>(shmem_warp_stream_ptr + BLOCK_COL_SIZE * i);
      auto gmem_i_ptr = reinterpret_cast<copy_t*>(src_gmem_warp_stream_ptr + n * i);
      gmem_i_ptr[lane_id] = smem_i_ptr[lane_id];
    }
    __syncthreads();
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

void gemm_v3(torch::Tensor& a, torch::Tensor& b, torch::Tensor& c) {
  cr::cr_assert(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(a.size(0) == c.size(0), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(b.size(1) == c.size(1), "%s:%d", __FILE__, __LINE__);
  cr::cr_assert(a.size(1) == b.size(0), "%s:%d", __FILE__, __LINE__);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  half* a_ptr = reinterpret_cast<half*>(a.data_ptr());
  half* b_ptr = reinterpret_cast<half*>(b.data_ptr());
  float* c_ptr = reinterpret_cast<float*>(c.data_ptr());
  unsigned m = a.size(0), k = a.size(1), n = b.size(1);
  unsigned block_num_m = m / BLOCK_ROW_SIZE;
  unsigned block_num_n = n / BLOCK_COL_SIZE;

  cudaDeviceProp device_prop;
  {
    cudaError_t ret = cudaGetDeviceProperties(&device_prop, 0);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
  }
  if (device_prop.sharedMemPerMultiprocessor <= 64 * 1024) {
    constexpr int CHUNK_K = 4;
    constexpr int SM_SIZE = MAX((M * BLOCK_ROW_TILES) * (N * BLOCK_COL_TILES) * sizeof(float),
        (M * BLOCK_ROW_TILES) * (CHUNK_K * K + SKEW_HALF) * 2 * sizeof(half));
    cudaError_t ret =
        cudaFuncSetAttribute(gemm_kernel_v3<CHUNK_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SIZE);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
    gemm_kernel_v3<CHUNK_K>
        <<<block_num_m * block_num_n, NUM_THREADS, SM_SIZE>>>(a_ptr, b_ptr, c_ptr, m, n, k, 1.f, 0.f);
  } else {
    constexpr int CHUNK_K = 8;
    constexpr int SM_SIZE = MAX((M * BLOCK_ROW_TILES) * (N * BLOCK_COL_TILES) * sizeof(float),
        (M * BLOCK_ROW_TILES) * (CHUNK_K * K + SKEW_HALF) * 2 * sizeof(half));
    cudaError_t ret =
        cudaFuncSetAttribute(gemm_kernel_v3<CHUNK_K>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_SIZE);
    if (ret != cudaSuccess) {
      throw CUDARuntimeError(ret);
    }
    gemm_kernel_v3<CHUNK_K>
        <<<block_num_m * block_num_n, NUM_THREADS, SM_SIZE>>>(a_ptr, b_ptr, c_ptr, m, n, k, 1.f, 0.f);
  }
}

}  // namespace cr
