#pragma once

#include <cstdint>

#include <cooperative_groups/memcpy_async.h>
#include <cuda_fp16.h>
#include <cuda/pipeline>

#include "./attention_dtypes.h"
#include "./attention_generic.cuh"
#include "./attention_utils.cuh"

namespace cr {

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
using namespace vllm;

template <class Vec, int N>
inline __device__ void cr_fma(float (&c)[N], float a, Vec b);

template <>
inline __device__ void cr_fma(float (&c)[2], float a, uint32_t b) {
  auto h0 = __half2float(__ushort_as_half(b & 0xFFFF));
  auto h1 = __half2float(__ushort_as_half((b >> 16) & 0xFFFF));
  c[0] += a * h0;
  c[1] += a * h1;
}

template <>
inline __device__ void cr_fma(float (&c)[4], float a, uint2 b) {
  auto h0 = __half2float(__ushort_as_half(b.x & 0xFFFF));
  auto h1 = __half2float(__ushort_as_half((b.x >> 16) & 0xFFFF));
  auto h2 = __half2float(__ushort_as_half(b.y & 0xFFFF));
  auto h3 = __half2float(__ushort_as_half((b.y >> 16) & 0xFFFF));
  c[0] += a * h0;
  c[1] += a * h1;
  c[2] += a * h2;
  c[3] += a * h3;
}

template <>
inline __device__ void cr_fma(float (&c)[8], float a, uint4 b) {
  auto h0 = __half2float(__ushort_as_half(b.x & 0xFFFF));
  auto h1 = __half2float(__ushort_as_half((b.x >> 16) & 0xFFFF));
  auto h2 = __half2float(__ushort_as_half(b.y & 0xFFFF));
  auto h3 = __half2float(__ushort_as_half((b.y >> 16) & 0xFFFF));
  auto h4 = __half2float(__ushort_as_half(b.z & 0xFFFF));
  auto h5 = __half2float(__ushort_as_half((b.z >> 16) & 0xFFFF));
  auto h6 = __half2float(__ushort_as_half(b.w & 0xFFFF));
  auto h7 = __half2float(__ushort_as_half((b.w >> 16) & 0xFFFF));
  c[0] += a * h0;
  c[1] += a * h1;
  c[2] += a * h2;
  c[3] += a * h3;
  c[4] += a * h4;
  c[5] += a * h5;
  c[6] += a * h6;
  c[7] += a * h7;
}

template <class vec_t, int N>
inline __device__ void cr_arr_to_vec_t(vec_t& dst, float (&src)[N]);

template <>
inline __device__ void cr_arr_to_vec_t(uint32_t& dst, float (&src)[2]) {
  float2 x;
  x.x = src[0];
  x.y = src[1];
  dst = float2_to_half2(x);
}

template <>
inline __device__ void cr_arr_to_vec_t(uint2& dst, float (&src)[4]) {
  float2 x;
  x.x = src[0];
  x.y = src[1];
  dst.x = float2_to_half2(x);
  float2 y;
  y.x = src[2];
  y.y = src[3];
  dst.y = float2_to_half2(y);
}

template <>
inline __device__ void cr_arr_to_vec_t(uint4& dst, float (&src)[8]) {
  float2 w;
  w.x = src[0];
  w.y = src[1];
  dst.w = float2_to_half2(w);
  float2 x;
  x.x = src[2];
  x.y = src[3];
  dst.x = float2_to_half2(x);
  float2 y;
  y.x = src[4];
  y.y = src[5];
  dst.y = float2_to_half2(y);
  float2 z;
  z.x = src[6];
  z.y = src[7];
  dst.z = float2_to_half2(z);
}

template <int N>
inline __device__ void print_vec(float (&v)[N]);

template <>
inline __device__ void print_vec(float (&v)[2]) {
  printf("%f %f\n", v[0], v[1]);
}

template <>
inline __device__ void print_vec(float (&v)[4]) {
  printf("%f %f %f %f\n", v[0], v[1], v[2], v[3]);
}

template <>
inline __device__ void print_vec(float (&v)[8]) {
  printf("%f %f %f %f %f %f %f %f\n", v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]);
}

template <typename vec_t>
inline __device__ void print_fp16(vec_t v);

template <>
inline __device__ void print_fp16(uint32_t v) {
  auto h0 = __half2float(__ushort_as_half(v & 0xFFFF));
  auto h1 = __half2float(__ushort_as_half(v >> 16 & 0xFFFF));
  printf("%f %f\n", h0, h1);
}

template <>
inline __device__ void print_fp16(uint2 v) {
  auto h0 = __half2float(__ushort_as_half(v.x & 0xFFFF));
  auto h1 = __half2float(__ushort_as_half(v.x >> 16 & 0xFFFF));
  auto h2 = __half2float(__ushort_as_half(v.y & 0xFFFF));
  auto h3 = __half2float(__ushort_as_half(v.y >> 16 & 0xFFFF));
  printf("%f %f %f %f\n", h0, h1, h2, h3);
}

template <>
inline __device__ void print_fp16(uint4 val) {
  auto h0 = __half2float(__ushort_as_half(val.x & 0xFFFF));
  auto h1 = __half2float(__ushort_as_half((val.x >> 16) & 0xFFFF));
  auto h2 = __half2float(__ushort_as_half(val.y & 0xFFFF));
  auto h3 = __half2float(__ushort_as_half((val.y >> 16) & 0xFFFF));
  auto h4 = __half2float(__ushort_as_half(val.z & 0xFFFF));
  auto h5 = __half2float(__ushort_as_half((val.z >> 16) & 0xFFFF));
  auto h6 = __half2float(__ushort_as_half(val.w & 0xFFFF));
  auto h7 = __half2float(__ushort_as_half((val.w >> 16) & 0xFFFF));
  printf("%f,%f,%f,%f,%f,%f,%f,%f\n", h0, h1, h2, h3, h4, h5, h6, h7);
}

struct Reduce {
  struct Max {
    static inline __device__ float apply(float a, float b) { return fmaxf(a, b); }
  };

  struct Add {
    static inline __device__ float apply(float a, float b) { return a + b; }
  };

  template <class Op, class T, int NUM_WARPS>
  static inline __device__ T reduce(T* red_smem, T sum) {
    // Decompose the thread index into warp / lane.
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // Compute the sum per warp.
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      sum = Op::apply(__shfl_xor_sync(uint32_t(-1), sum, mask), sum);
    }

    // Warp leaders store the data to shared memory.
    if (lane == 0) {
      red_smem[warp] = sum;
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    // The warps compute the final sums.
    if (lane < NUM_WARPS) {
      sum = red_smem[lane];
    }

    // Parallel reduction inside the warp.
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
      sum = Op::apply(__shfl_xor_sync(uint32_t(-1), sum, mask), sum);
    }

    // Broadcast to other threads.
    return __shfl_sync(uint32_t(-1), sum, 0);
  }

  template <class Op, class T, int NUM_WARPS, int N>
  static inline __device__ void reduceToWarp0(T* red_smem, T (&v)[N]) {
    const int thread_idx = threadIdx.x;
    const int warp_idx = thread_idx / WARP_SIZE;
    const int lane = thread_idx % WARP_SIZE;
    constexpr int NUM_ELEMENTS_PER_WARP = N * WARP_SIZE;

#pragma unroll
    for (int i = NUM_WARPS; i > 1; i /= 2) {
      int mid = i / 2;
      // Upper warps write to shared memory.
      if (warp_idx >= mid && warp_idx < i) {
        T* dst = &red_smem[(warp_idx - mid) * NUM_ELEMENTS_PER_WARP];
#pragma unroll
        for (int i = 0; i < N; ++i) {
          const int row_idx = i * WARP_SIZE + lane;
          dst[row_idx] = v[i];
        }
      }
      __syncthreads();

      // Lower warps update the output.
      if (warp_idx < mid) {
        const T* src = &red_smem[warp_idx * NUM_ELEMENTS_PER_WARP];
#pragma unroll
        for (int i = 0; i < N; ++i) {
          const int row_idx = i * WARP_SIZE + lane;
          // v[i] += src[row_idx];
          v[i] = Op::apply(v[i], src[row_idx]);
        }
      }
      __syncthreads();
    }
  }
};

template <int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct Softmax {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  static inline __device__ float doSoftmax(float* logits, float* red_smem, const float qk_max, const int context_len) {
    const int thread_idx = threadIdx.x;

    float exp_sum = 0.f;
    for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
      float val = __expf(logits[i] - qk_max);
      logits[i] = val;
      exp_sum += val;
    }
    exp_sum = Reduce::reduce<Reduce::Add, float, NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

    // Compute softmax.
    const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
    for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
      logits[i] *= inv_sum;
    }
    __syncthreads();

    return exp_sum;
  }
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct LoadQAndGemmQk_CR1 {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int N = HEAD_SIZE / WARP_SIZE;
  using vec_t = typename Vec<scalar_t, N>::Type;

  inline __device__ LoadQAndGemmQk_CR1(const scalar_t* q, scalar_t* smem_q, const scalar_t* k_cache,
      const int32_t* block_table, float* logits, float* red_smem, const int seq_idx, const int head_idx,
      const int kv_head_idx, const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int context_len, const int num_blocks, const float scale, const float alibi_slope, const int kv_offset = 0)
      : q(q),
        k_cache(k_cache),
        block_table(block_table),
        logits(logits),
        red_smem(red_smem),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        q_stride(q_stride),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        num_blocks(num_blocks),
        scale(scale),
        alibi_slope(alibi_slope),
        kv_offset(kv_offset),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE) {}

  inline __device__ void loadQ() {
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
    // NOTE: Here we compare 2 methods: 1. All threads load data to registers and no sync.
    //                                  2. Warp 0 load data to shared memory and then sync to all threads.
    //       The former increases the access to global memory, and the latter adds a thread synchronization overhead.
    //       Experiments show that the former is faster.
    q_vec = *reinterpret_cast<const vec_t*>(q_ptr + lane * N);
  }

  inline __device__ float gemmQk() {
    float qk_max = -FLT_MAX;
    float qk[BLOCK_SIZE];
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];

#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride +
                                physical_block_offset * HEAD_SIZE;
        const int vec_idx = lane;
        vec_t k_vec = *reinterpret_cast<const vec_t*>(k_ptr + vec_idx * N);
        using A_vec = typename FloatVec<vec_t>::Type;
        A_vec qk_vec = mul<A_vec, vec_t, vec_t>(q_vec, k_vec);
        qk[physical_block_offset] = sum(qk_vec);
      }
#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        // NOTE: We handle this loop independently because __shfl will cause an extra synchronization of threads within
        //       the warp.
        float token_qk = qk[physical_block_offset];
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
          token_qk += __shfl_xor_sync(uint32_t(-1), token_qk, mask);
        }
        qk[physical_block_offset] = token_qk;
      }

      if (lane < BLOCK_SIZE) {
        const int token_idx = block_idx * BLOCK_SIZE + lane;
        float qk_ = qk[lane];
        // NOTE: Each element add a constant value before softmax will not change the result.
        // qk_ += alibi_slope != 0 ? alibi_slope * (token_idx - context_len + 1) : 0;
        qk_ += alibi_slope * (token_idx + kv_offset);
        qk_ *= scale;
        qk_ = token_idx < context_len ? qk_ : -FLT_MAX;
        logits[token_idx] = qk_;
        qk_max = fmaxf(qk_max, qk_);
      }
    }
    // NOTE: qk_max only impact the accuracy of softmax, it will not change the result theoretically.
    // NOTE: The accuracy of CUDA's exp is lower than normal c++ source file, so we should always sub the max value
    //       before softmax.
    qk_max = Reduce::reduce<Reduce::Max, float, NUM_WARPS>(red_smem, qk_max);
    return qk_max;
  }

  const scalar_t* q;
  const scalar_t* k_cache;
  const int32_t* block_table;
  float* logits;
  float* red_smem;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;
  const int q_stride;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;
  const int num_blocks;

  const float scale;
  const float alibi_slope;
  const int kv_offset;

  const int thread_idx;
  const int warp_idx;
  const int lane;

  vec_t q_vec;
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct LoadQAndGemmQk_CR2 {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  using vec_t = uint4;
  static constexpr int N = sizeof(vec_t) / sizeof(scalar_t);
  static constexpr int GROUP_SIZE = HEAD_SIZE / N;
  static constexpr int NUM_GROUPS = WARP_SIZE / GROUP_SIZE;
  static_assert(NUM_GROUPS == 1 || NUM_GROUPS == 2 || NUM_GROUPS == 4);
  static constexpr int NUM_ITERS = BLOCK_SIZE / NUM_GROUPS;

  inline __device__ LoadQAndGemmQk_CR2(const scalar_t* q, scalar_t* smem_q, const scalar_t* k_cache,
      const int32_t* block_table, float* logits, float* red_smem, const int seq_idx, const int head_idx,
      const int kv_head_idx, const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int context_len, const int num_blocks, const float scale, const float alibi_slope)
      : q(q),
        k_cache(k_cache),
        block_table(block_table),
        logits(logits),
        red_smem(red_smem),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        q_stride(q_stride),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        num_blocks(num_blocks),
        scale(scale),
        alibi_slope(alibi_slope),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE),
        group_idx(lane / GROUP_SIZE),
        group_offset(lane % GROUP_SIZE) {}

  inline __device__ void loadQ() {
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
    q_vec = *reinterpret_cast<const vec_t*>(q_ptr + group_offset * N);
  }

  inline __device__ float gemmQk() {
    using kvec_t = uint4;

    float qk_max = -FLT_MAX;
    float qk[NUM_ITERS];
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
#pragma unroll
      for (int iter = 0; iter < NUM_ITERS; ++iter) {
        const int physical_block_offset = iter * NUM_GROUPS + group_idx;
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride +
                                physical_block_offset * HEAD_SIZE;
        vec_t k_vec = *reinterpret_cast<const vec_t*>(k_ptr + group_offset * N);
        using A_vec = typename FloatVec<vec_t>::Type;
        A_vec qk_vec = mul<A_vec, vec_t, vec_t>(q_vec, k_vec);
        qk[physical_block_offset] = sum(qk_vec);
      }
#pragma unroll
      for (int iter = 0; iter < NUM_ITERS; ++iter) {
        const int physical_block_offset = iter * NUM_GROUPS + group_idx;
        float token_qk = qk[physical_block_offset];
#pragma unroll
        for (int mask = GROUP_SIZE / 2; mask >= 1; mask /= 2) {
          token_qk += __shfl_xor_sync(uint32_t(-1), token_qk, mask);
        }
        qk[physical_block_offset] = token_qk;
      }
      if (group_offset < NUM_ITERS) {
        const int physical_block_offset = group_offset * NUM_GROUPS + group_idx;
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
        float qk_ = qk[physical_block_offset];
        qk_ += alibi_slope * token_idx;
        qk_ *= scale;
        qk_ = token_idx < context_len ? qk_ : -FLT_MAX;
        logits[token_idx] = qk_;
        qk_max = fmaxf(qk_max, qk_);
      }
    }
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= GROUP_SIZE; mask /= 2) {
      qk_max = fmaxf(__shfl_xor_sync(uint32_t(-1), qk_max, mask), qk_max);
    }

    // NOTE: qk_max only impact the accuracy of softmax, it will not change the result theoretically.
    // NOTE: The accuracy of CUDA's exp is lower than normal c++ source file, so we should always sub the max value
    //       before softmax.
    qk_max = Reduce::reduce<Reduce::Max, float, NUM_WARPS>(red_smem, qk_max);
    return qk_max;
  }

  const scalar_t* q;
  const scalar_t* k_cache;
  const int32_t* block_table;
  float* logits;
  float* red_smem;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;
  const int q_stride;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;
  const int num_blocks;

  const float scale;
  const float alibi_slope;

  const int thread_idx;
  const int warp_idx;
  const int lane;
  const int group_idx;
  const int group_offset;

  vec_t q_vec;
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct LoadQAndGemmQk_CR3 {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int N = HEAD_SIZE / WARP_SIZE;

  inline __device__ LoadQAndGemmQk_CR3(const scalar_t* q, scalar_t* smem_q, const scalar_t* k_cache,
      const int32_t* block_table, float* logits, float* red_smem, const int seq_idx, const int head_idx,
      const int kv_head_idx, const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int context_len, const int num_blocks, const float scale, const float alibi_slope)
      : q(q),
        k_cache(k_cache),
        block_table(block_table),
        logits(logits),
        red_smem(red_smem),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        q_stride(q_stride),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        num_blocks(num_blocks),
        scale(scale),
        alibi_slope(alibi_slope),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE) {}

  inline __device__ void loadQ() {
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      q_vec[i] = *(q_ptr + i * WARP_SIZE + lane);
    }
  }

  inline __device__ float gemmQk() {
    float qk_max = -FLT_MAX;
    float qk[BLOCK_SIZE];

    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
#pragma unroll
      for (int i = 0; i < BLOCK_SIZE; ++i) {
        qk[i] = 0.f;
      }

      const int physical_block_number = block_table[block_idx];

#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride +
                                physical_block_offset * HEAD_SIZE;
#pragma unroll
        for (int i = 0; i < N; ++i) {
          scalar_t k_ = *(k_ptr + i * WARP_SIZE + lane);
          using A_vec = typename FloatVec<scalar_t>::Type;
          qk[physical_block_offset] += mul<A_vec, scalar_t, scalar_t>(q_vec[i], k_);
        }
      }
#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        float token_qk = qk[physical_block_offset];
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
          token_qk += __shfl_xor_sync(uint32_t(-1), token_qk, mask);
        }
        qk[physical_block_offset] = token_qk;
      }

      if (lane < BLOCK_SIZE) {
        const int token_idx = block_idx * BLOCK_SIZE + lane;
        float qk_ = qk[lane];
        // NOTE: Each element add a constant value before softmax will not change the result.
        // qk_ += alibi_slope != 0 ? alibi_slope * (token_idx - context_len + 1) : 0;
        qk_ += alibi_slope * token_idx;
        qk_ *= scale;
        qk_ = token_idx < context_len ? qk_ : -FLT_MAX;
        logits[token_idx] = qk_;
        qk_max = fmaxf(qk_max, qk_);
      }
    }

    // NOTE: qk_max only impact the accuracy of softmax, it will not change the result theoretically.
    // NOTE: The accuracy of CUDA's exp is lower than normal c++ source file, so we should always sub the max value
    //       before softmax.
    qk_max = Reduce::reduce<Reduce::Max, float, NUM_WARPS>(red_smem, qk_max);
    return qk_max;
  }

  const scalar_t* q;
  const scalar_t* k_cache;
  const int32_t* block_table;
  float* logits;
  float* red_smem;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;
  const int q_stride;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;
  const int num_blocks;

  const float scale;
  const float alibi_slope;

  const int thread_idx;
  const int warp_idx;
  const int lane;

  scalar_t q_vec[N];
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct LoadQAndGemmQk_CR4 {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int N = HEAD_SIZE / WARP_SIZE;
  static constexpr int NUM_STAGES = 8;
  using vec_t = typename Vec<scalar_t, N>::Type;

  inline __device__ LoadQAndGemmQk_CR4(const scalar_t* q, vec_t* smem_k_vecs, const scalar_t* k_cache,
      const int32_t* block_table, float* logits, float* red_smem, const int seq_idx, const int head_idx,
      const int kv_head_idx, const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int context_len, const int num_blocks, const float scale, const float alibi_slope, const int kv_offset = 0)
      : q(q),
        k_vecs(reinterpret_cast<decltype(k_vecs)>(*smem_k_vecs)),
        k_cache(k_cache),
        block_table(block_table),
        logits(logits),
        red_smem(red_smem),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        q_stride(q_stride),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        num_blocks(num_blocks),
        scale(scale),
        alibi_slope(alibi_slope),
        kv_offset(kv_offset),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE) {}

  inline __device__ void loadQ() {
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
    // NOTE: Here we compare 2 methods: 1. All threads load data to registers and no sync.
    //                                  2. Warp 0 load data to shared memory and then sync to all threads.
    //       The former increases the access to global memory, and the latter adds a thread synchronization overhead.
    //       Experiments show that the former is faster.
    q_vec = *reinterpret_cast<const vec_t*>(q_ptr + lane * N);
  }

  inline __device__ float gemmQk() {
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    float qk_max = -FLT_MAX;
    float qk[BLOCK_SIZE];

    if (warp_idx < num_blocks) {
      const int physical_block_number = block_table[warp_idx];
      const vec_t* k_base_ptr = reinterpret_cast<const vec_t*>(
          k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);
#pragma unroll
      for (int i = 0; i < NUM_STAGES - 1; ++i) {
        pipeline.producer_acquire();
        cuda::memcpy_async(
            &k_vecs[NUM_STAGES * warp_idx + i][lane], k_base_ptr + i * WARP_SIZE + lane, sizeof(vec_t), pipeline);
        pipeline.producer_commit();
      }
    }

    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
      const vec_t* k_base_ptr = reinterpret_cast<const vec_t*>(
          k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);

#pragma unroll
      for (int physical_block_offset = NUM_STAGES - 1; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        const vec_t* k_ptr = k_base_ptr + physical_block_offset * WARP_SIZE;

        const int copy_stage_idx = physical_block_offset % NUM_STAGES;
        pipeline.producer_acquire();
        cuda::memcpy_async(
            &k_vecs[NUM_STAGES * warp_idx + copy_stage_idx][lane], k_ptr + lane, sizeof(vec_t), pipeline);
        pipeline.producer_commit();

        const int load_stage_idx = (physical_block_offset + 1) % NUM_STAGES;
        pipeline.consumer_wait();
        vec_t k_vec = k_vecs[NUM_STAGES * warp_idx + load_stage_idx][lane];
        pipeline.consumer_release();

        using A_vec = typename FloatVec<vec_t>::Type;
        A_vec qk_vec = mul<A_vec, vec_t, vec_t>(q_vec, k_vec);
        qk[physical_block_offset - NUM_STAGES + 1] = sum(qk_vec);
      }

      const vec_t* next_k_base_ptr = nullptr;
      if (block_idx + NUM_WARPS < num_blocks) {
        const int next_physical_block_number = block_table[block_idx + NUM_WARPS];
        next_k_base_ptr = reinterpret_cast<const vec_t*>(
            k_cache + next_physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);
      }

#pragma unroll
      for (int i = 0; i < NUM_STAGES - 1; ++i) {
        if (next_k_base_ptr) {
          pipeline.producer_acquire();
          cuda::memcpy_async(&k_vecs[NUM_STAGES * warp_idx + i][lane], next_k_base_ptr + i * WARP_SIZE + lane,
              sizeof(vec_t), pipeline);
          pipeline.producer_commit();
        }

        const int physical_block_offset = BLOCK_SIZE + i;

        const int load_stage_idx = (physical_block_offset + 1) % NUM_STAGES;
        pipeline.consumer_wait();
        vec_t k_vec = k_vecs[NUM_STAGES * warp_idx + load_stage_idx][lane];
        pipeline.consumer_release();

        using A_vec = typename FloatVec<vec_t>::Type;
        A_vec qk_vec = mul<A_vec, vec_t, vec_t>(q_vec, k_vec);
        qk[physical_block_offset - NUM_STAGES + 1] = sum(qk_vec);
      }

#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        // NOTE: We handle this loop independently because __shfl will cause an extra synchronization of threads within
        //       the warp.
        float token_qk = qk[physical_block_offset];
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
          token_qk += __shfl_xor_sync(uint32_t(-1), token_qk, mask);
        }
        qk[physical_block_offset] = token_qk;
      }

      if (lane < BLOCK_SIZE) {
        const int token_idx = block_idx * BLOCK_SIZE + lane;
        float qk_ = qk[lane];
        // NOTE: Each element add a constant value before softmax will not change the result.
        // qk_ += alibi_slope != 0 ? alibi_slope * (token_idx - context_len + 1) : 0;
        qk_ += alibi_slope * (token_idx + kv_offset);
        qk_ *= scale;
        qk_ = token_idx < context_len ? qk_ : -FLT_MAX;
        logits[token_idx] = qk_;
        qk_max = fmaxf(qk_max, qk_);
      }
    }
    // NOTE: qk_max only impact the accuracy of softmax, it will not change the result theoretically.
    // NOTE: The accuracy of CUDA's exp is lower than normal c++ source file, so we should always sub the max value
    //       before softmax.
    qk_max = Reduce::reduce<Reduce::Max, float, NUM_WARPS>(red_smem, qk_max);
    return qk_max;
  }

  const scalar_t* q;
  const scalar_t* k_cache;
  const int32_t* block_table;
  float* logits;
  float* red_smem;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;
  const int q_stride;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;
  const int num_blocks;

  const float scale;
  const float alibi_slope;
  const int kv_offset;

  const int thread_idx;
  const int warp_idx;
  const int lane;

  vec_t q_vec;
  vec_t (&k_vecs)[NUM_STAGES * NUM_WARPS][WARP_SIZE];
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct GemmPvAndStoreO_CR1 {
  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int N = HEAD_SIZE / WARP_SIZE;

  inline __device__ GemmPvAndStoreO_CR1(scalar_t* out, const scalar_t* v_cache, const int32_t* block_table,
      float* logits, const int seq_idx, const int head_idx, const int kv_head_idx, const int num_heads,
      const int num_blocks, const int kv_block_stride, const int kv_head_stride, const int context_len)
      : out(out),
        v_cache(v_cache),
        block_table(block_table),
        logits(logits),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        num_heads(num_heads),
        num_blocks(num_blocks),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      accs[i] = 0.f;
    }
  }

  inline __device__ void gemmPv() {
    using vec_t = typename Vec<scalar_t, N>::Type;
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
      const half* v_ptr = reinterpret_cast<const half*>(
          v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);
#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
        // NOTE: Out-of-bounds data has been processed previously, and its value is zero.
        // NOTE: Experiments have shown that in this case, it will be a little faster to remove the conditional judgment
        float token_logit = token_idx < context_len ? logits[token_idx] : 0.f;
        // float token_logit = logits[token_idx];
        const int offset1 = physical_block_offset * HEAD_SIZE;
        const int offset2 = lane * N;
        vec_t v = *reinterpret_cast<const vec_t*>(v_ptr + offset1 + offset2);
        cr_fma<vec_t, N>(accs, token_logit, v);
      }
    }
  }

  inline __device__ void storeO() {
    using vec_t = typename Vec<scalar_t, N>::Type;
    __syncthreads();
    // logits had completed its work.
    Reduce::reduceToWarp0<Reduce::Add, float, NUM_WARPS, N>(logits, accs);
    if (warp_idx == 0) {
      scalar_t* out_ptr = out;
      cr_arr_to_vec_t(*reinterpret_cast<vec_t*>(out_ptr + lane * N), accs);
    }
  }

  scalar_t* out;
  const scalar_t* v_cache;
  const int32_t* block_table;
  float* logits;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;

  const int num_heads;
  const int num_blocks;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;

  const int thread_idx;
  const int warp_idx;
  const int lane;
  float accs[N];
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct GemmPvAndStoreO_CR2 {
  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  using vec_t = uint4;
  static constexpr int VEC_SIZE = sizeof(vec_t) / sizeof(scalar_t);
  static constexpr int NUM_ELEMENTS_PER_THREAD = VEC_SIZE;

  // one thread group deal with one line.
  static constexpr int THREAD_GROUP_SIZE = HEAD_SIZE / VEC_SIZE;
  static constexpr int NUM_GROUPS = WARP_SIZE / THREAD_GROUP_SIZE;
  static_assert(NUM_GROUPS == 1 || NUM_GROUPS == 2 || NUM_GROUPS == 4);
  static constexpr int NUM_ITERS_PER_BLOCK = BLOCK_SIZE / NUM_GROUPS;

  inline __device__ GemmPvAndStoreO_CR2(scalar_t* out, const scalar_t* v_cache, const int32_t* block_table,
      float* logits, const int seq_idx, const int head_idx, const int kv_head_idx, const int num_heads,
      const int num_blocks, const int kv_block_stride, const int kv_head_stride, const int context_len)
      : out(out),
        v_cache(v_cache),
        block_table(block_table),
        logits(logits),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        num_heads(num_heads),
        num_blocks(num_blocks),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE),
        group_idx(lane / THREAD_GROUP_SIZE),
        group_offset(lane % THREAD_GROUP_SIZE) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      accs[i] = 0.f;
    }
  }

  inline __device__ void gemmPv() {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      accs[i] = 0.f;
    }
#pragma unroll
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
      const half* v_ptr = reinterpret_cast<const half*>(
          v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);
#pragma unroll
      for (int iter = 0; iter < NUM_ITERS_PER_BLOCK; ++iter) {
        const int physical_block_offset = group_idx + iter * NUM_GROUPS;
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
        float token_logit = token_idx < context_len ? logits[token_idx] : 0.f;
        const int offset1 = physical_block_offset * HEAD_SIZE;
        const int offset2 = group_offset * VEC_SIZE;
        vec_t v = *reinterpret_cast<const vec_t*>(v_ptr + offset1 + offset2);
        cr_fma<vec_t, VEC_SIZE>(accs, token_logit, v);
      }
    }
  }

  inline __device__ void storeO() {
    /**
     * Example: HEAD_SIZE = 64
     *
     * From
     *         |----offset0----| ... |----offset7----|
     * group 0 |0 1 2 3 4 5 6 7|
     * group 1 |0 1 2 3 4 5 6 7|
     * group 2 |0 1 2 3 4 5 6 7|
     * group 3 |0 1 2 3 4 5 6 7|
     *
     * To
     *         |----offset0----| ... |----offset7----|
     * group 0 |0 1            |
     * group 1 |    2 3        |
     * group 2 |        4 5    |
     * group 3 |            7 8|
     */
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        accs[i] += __shfl_xor_sync(uint32_t(-1), accs[i], mask);
      }
    }
    constexpr int N = HEAD_SIZE / WARP_SIZE;
    using vec_t = typename Vec<scalar_t, N>::Type;
    float accs_to[N];
#pragma unroll
    for (int i = 0; i < N; ++i) {
      accs_to[i] = accs[group_idx * N + i];
    }

    Reduce::reduceToWarp0<Reduce::Add, float, NUM_WARPS, N>(logits, accs_to);

    if (warp_idx == 0) {
      scalar_t* out_ptr = out;
      const int offset1 = group_idx * N;
      const int offset2 = group_offset * VEC_SIZE;
      cr_arr_to_vec_t(*reinterpret_cast<vec_t*>(out_ptr + offset1 + offset2), accs_to);
    }
  }

  scalar_t* out;
  const scalar_t* v_cache;
  const int32_t* block_table;
  float* logits;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;

  const int num_heads;
  const int num_blocks;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;

  const int thread_idx;
  const int warp_idx;
  const int lane;
  const int group_idx;
  const int group_offset;
  float accs[NUM_ELEMENTS_PER_THREAD];
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct GemmPvAndStoreO_CR3 {
  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int N = HEAD_SIZE / WARP_SIZE;

  inline __device__ GemmPvAndStoreO_CR3(scalar_t* out, const scalar_t* v_cache, const int32_t* block_table,
      float* logits, const int seq_idx, const int head_idx, const int kv_head_idx, const int num_heads,
      const int num_blocks, const int kv_block_stride, const int kv_head_stride)
      : out(out),
        v_cache(v_cache),
        block_table(block_table),
        logits(logits),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        num_heads(num_heads),
        num_blocks(num_blocks),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
      accs[i] = 0.f;
    }
  }

  inline __device__ void gemmPv() {
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
      const half* v_ptr = reinterpret_cast<const half*>(
          v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);
#pragma unroll
      for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
        float token_logit = logits[token_idx];
        const int offset1 = physical_block_offset * HEAD_SIZE;
#pragma unroll
        for (int i = 0; i < N; ++i) {
          const int offset2 = WARP_SIZE * i + lane;
          half v = *(v_ptr + offset1 + offset2);
          accs[i] += __half2float(v) * token_logit;
        }
      }
    }
  }

  inline __device__ void storeO() {
    using vec_t = typename Vec<scalar_t, N>::Type;
    // logits had completed its work.
    Reduce::reduceToWarp0<Reduce::Add, float, NUM_WARPS, N>(logits, accs);
    if (warp_idx == 0) {
#pragma unroll
      for (int i = 0; i < N; ++i) {
        scalar_t* out_ptr = out;
        from_float(*(out_ptr + i * WARP_SIZE + lane), accs[i]);
      }
    }
  }

  scalar_t* out;
  const scalar_t* v_cache;
  const int32_t* block_table;
  float* logits;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;

  const int num_heads;
  const int num_blocks;
  const int kv_block_stride;
  const int kv_head_stride;

  const int thread_idx;
  const int warp_idx;
  const int lane;
  float accs[N];
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct LoadQAndGemmQk_VLLM1 {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  static constexpr int NUM_THREAD_GROUPS_PER_WARP = WARP_SIZE / THREAD_GROUP_SIZE;
  static constexpr int NUM_TOKENS_PER_THREAD_GROUP = BLOCK_SIZE / NUM_THREAD_GROUPS_PER_WARP;
  static constexpr int NUM_THREAD_GROUPS = NUM_WARPS * NUM_THREAD_GROUPS_PER_WARP;
  static constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  static constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  static constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;
  static constexpr int x = 16 / sizeof(scalar_t);

  inline __device__ LoadQAndGemmQk_VLLM1(const scalar_t* q, scalar_t* smem_q, const scalar_t* k_cache,
      const int32_t* block_table, float* logits, float* red_smem, const int seq_idx, const int head_idx,
      const int kv_head_idx, const int q_stride, const int kv_block_stride, const int kv_head_stride,
      const int context_len, const int num_blocks, const float scale, const float alibi_slope, const int kv_offset = 0)
      : q(q),
        q_vecs(reinterpret_cast<decltype(q_vecs)>(*smem_q)),
        k_cache(k_cache),
        block_table(block_table),
        logits(logits),
        red_smem(red_smem),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        q_stride(q_stride),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        num_blocks(num_blocks),
        scale(scale),
        alibi_slope(alibi_slope),
        kv_offset(kv_offset),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE),
        thread_group_idx(thread_idx / THREAD_GROUP_SIZE),
        thread_group_offset(thread_idx % THREAD_GROUP_SIZE),
        thread_group_idx_in_warp(lane / THREAD_GROUP_SIZE) {}

  inline __device__ void loadQ() {
    const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
      const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
      q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
    }
    __syncthreads();
  }

  inline __device__ float gemmQk() {
    float qk_max = -FLT_MAX;
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
      for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
        const int physical_block_offset = thread_group_idx_in_warp + i * WARP_SIZE;
        const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
        K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
        for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
          const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride +
                                  physical_block_offset * x;
          const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
          const int offset1 = (vec_idx * VEC_SIZE) / x;
          const int offset2 = (vec_idx * VEC_SIZE) % x;
          k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }

        float qk = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
        qk += (alibi_slope != 0) ? alibi_slope * (kv_offset + token_idx) : 0;
        qk *= scale;

        if (thread_group_offset == 0) {
          const bool mask = token_idx >= context_len;
          logits[token_idx] = mask ? 0.f : qk;
          qk_max = mask ? qk_max : fmaxf(qk_max, qk);
        }
      }
    }
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
      qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
    if (lane == 0) {
      red_smem[warp_idx] = qk_max;
    }
    __syncthreads();
    qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
      qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
    }
    // Broadcast the max qk value to all threads.
    qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

    return qk_max;
  }

  const scalar_t* q;
  const scalar_t* k_cache;
  const int32_t* block_table;
  float* logits;
  float* red_smem;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;
  const int q_stride;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;
  const int num_blocks;

  const float scale;
  const float alibi_slope;
  const int kv_offset;

  const int thread_idx;
  const int warp_idx;
  const int lane;

  const int thread_group_idx;
  const int thread_group_offset;
  const int thread_group_idx_in_warp;

  Q_vec (&q_vecs)[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
};

template <class scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
struct GemmPvAndStoreO_VLLM1 {
  static constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  static constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;
  static constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  static constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  static constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  inline __device__ GemmPvAndStoreO_VLLM1(scalar_t* out, const scalar_t* v_cache, const int32_t* block_table,
      float* logits, const int seq_idx, const int head_idx, const int kv_head_idx, const int num_heads,
      const int num_blocks, const int kv_block_stride, const int kv_head_stride, const int context_len)
      : out(out),
        v_cache(v_cache),
        block_table(block_table),
        logits(logits),
        seq_idx(seq_idx),
        head_idx(head_idx),
        kv_head_idx(kv_head_idx),
        num_heads(num_heads),
        num_blocks(num_blocks),
        kv_block_stride(kv_block_stride),
        kv_head_stride(kv_head_stride),
        context_len(context_len),
        thread_idx(threadIdx.x),
        warp_idx(thread_idx / WARP_SIZE),
        lane(thread_idx % WARP_SIZE) {
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      accs[i] = 0.f;
    }
  }

  inline __device__ void gemmPv() {
    scalar_t zero_value;
    zero(zero_value);
    for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
      const int physical_block_number = block_table[block_idx];
      const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      L_vec logits_vec;
      from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx));

      const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE) {
          const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
          V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
          if (block_idx == num_blocks - 1) {
            scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
            for (int j = 0; j <= V_VEC_SIZE; j++) {
              v_vec_ptr[j] = token_idx + j < context_len ? v_vec_ptr[j] : zero_value;
            }
          }
          accs[i] += dot(logits_vec, v_vec);
        }
      }
    }

#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      float acc = accs[i];
#pragma unroll
      for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
        acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
      }
      accs[i] = acc;
    }
    __syncthreads();
  }

  inline __device__ void storeO() {
    // logits had completed its work.
    Reduce::reduceToWarp0<Reduce::Add, float, NUM_WARPS, NUM_ROWS_PER_THREAD>(logits, accs);
    if (warp_idx == 0) {
      scalar_t* out_ptr = out;
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          from_float(*(out_ptr + row_idx), accs[i]);
        }
      }
    }
  }

  scalar_t* out;
  const scalar_t* v_cache;
  const int32_t* block_table;
  float* logits;

  const int seq_idx;
  const int head_idx;
  const int kv_head_idx;

  const int num_heads;
  const int num_blocks;
  const int kv_block_stride;
  const int kv_head_stride;
  const int context_len;

  const int thread_idx;
  const int warp_idx;
  const int lane;

  float accs[NUM_ROWS_PER_THREAD];
};

}  // namespace cr
