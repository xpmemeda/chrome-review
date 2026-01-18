#include <ATen/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <algorithm>
#include <cstdint>
#include <cub/block/block_reduce.cuh>
#include <iostream>

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

using cutlass::Array;
using cutlass::half_t;

template <typename T, int VEC_SIZE>
struct Vec {
  using Type = Array<T, VEC_SIZE>;
};

template <typename VecT>
struct FloatVec;

template <int N>
struct FloatVec<Array<half_t, N>> {
  using Type = Array<float, N>;
};

template <typename Acc, typename A, typename B>
CUTLASS_DEVICE Acc mul(A const& a, B const& b) {
  Acc result;
#pragma unroll
  for (int i = 0; i < A::kElements; ++i) {
    result[i] = float(a[i]) * float(b[i]);
  }
  return result;
}

template <typename VecA, typename VecB, typename Acc>
CUTLASS_DEVICE Acc fma(VecA const& a, VecB const& b, Acc const& c) {
  Acc r;
#pragma unroll
  for (int i = 0; i < VecA::kElements; ++i) {
    r[i] = float(a[i]) * float(b[i]) + c[i];
  }
  return r;
}

template <typename Acc>
CUTLASS_DEVICE float sum(Acc const& a) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < Acc::kElements; ++i) {
    s += a[i];
  }
  return s;
}

template <int N>
CUTLASS_DEVICE float dot(Array<half_t, N> const& a, Array<half_t, N> const& b) {
  float s = 0.f;
#pragma unroll
  for (int i = 0; i < N; i++) {
    s += float(a[i]) * float(b[i]);
  }
  return s;
}

CUTLASS_DEVICE void from_float(half_t& dst, const float& src) {
  cutlass::NumericConverter<half_t, float> cvt;
  dst = cvt(src);
}

template <int N>
CUTLASS_DEVICE void from_float(Array<half_t, N>& dst, const Array<float, N>& src) {
  cutlass::NumericConverter<half_t, float> cvt;
#pragma unroll
  for (int i = 0; i < N; i++) {
    dst[i] = cvt(src[i]);
  }
}

CUTLASS_DEVICE void zero(half_t& x) { x = half_t(0); }

namespace vllm {

template <int THREAD_GROUP_SIZE, typename Vec, int N>
CUTLASS_DEVICE float qk_dot_(Vec const (&q)[N], Vec const (&k)[N]) {
  using Acc = Array<float, Vec::kElements>;

  Acc acc;

#pragma unroll
  for (int i = 0; i < Vec::kElements; i++) {
    acc[i] = 0.f;
  }

#pragma unroll
  for (int i = 0; i < N; ++i) {
#pragma unroll
    for (int j = 0; j < Vec::kElements; ++j) {
      acc[j] += float(q[i][j]) * float(k[i][j]);
    }
  }

  float qk = sum(acc);

#pragma unroll
  for (int mask = THREAD_GROUP_SIZE / 2; mask >= 1; mask /= 2) {
    qk += __shfl_xor_sync(0xffffffff, qk, mask);
  }

  return qk;
}

template <typename T, int THREAD_GROUP_SIZE>
struct Qk_dot {
  template <typename Vec, int N>
  static inline __device__ float dot(const Vec (&q)[N], const Vec (&k)[N]) {
    return qk_dot_<THREAD_GROUP_SIZE>(q, k);
  }
};

// Utility function for attention softmax.
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

  // Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
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
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

// Grid: (num_heads, num_seqs). One "BLOCK" works on one "SEQ-HEAD".
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(  //
    scalar_t* __restrict__ out,                           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,                       // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,                 // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,                 // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_heads_kv,                               //
    const float scale,                                    //
    const int32_t* __restrict__ block_tables,             // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,             // [num_seqs]
    const int max_num_blocks_per_seq,                     //
    const float* __restrict__ alibi_slopes,               // [num_heads]
    const int q_stride,                                   //
    const int kv_block_stride,                            //
    const int kv_head_stride                              //
) {
  static_assert(WARP_SIZE % BLOCK_SIZE == 0 || BLOCK_SIZE % WARP_SIZE == 0);
  static_assert(NUM_THREADS % WARP_SIZE == 0);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  // One "WARP" works on one kvcache block. Each "WARP" is divided into multiple "THREAD GROUP"s.
  //
  // One "THREAD GROUP" works on one or more "TOKEN"s in a kvcache block.
  // WARP_SIZE > BLOCK_SIZE:
  //    Multiple threads in "THREAD_GROUP" works on one "TOKEN".
  // WARP_SIZE < BLOCK_SIZE:
  //    "THREAD_GROUP" only contains one thread. One "THREAD_GROUP" works on multiple "TOKEN"s.
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS_PER_WARP = WARP_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = BLOCK_SIZE / NUM_THREAD_GROUPS_PER_WARP;
  constexpr int NUM_THREAD_GROUPS = NUM_WARPS * NUM_THREAD_GROUPS_PER_WARP;

  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / (num_heads / num_heads_kv);
  const int seq_idx = blockIdx.y;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // NOTE: 16 bytes is the cache line size.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  const int thread_group_idx_in_warp = lane / THREAD_GROUP_SIZE;

  // Load the query to shared memory (all threads).
  //
  // Each thread in a thread group has a different part of the query.
  // For example, if the the thread group size is 4, then the first thread in the group
  // has 0, 4, 8, ... th vectors of the query, and the second thread has 1, 5, 9, ...
  // th vectors of the query, and so on.
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];
#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // Each thread group fetches x elements from the key at a time.
  constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
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

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.

      // 这里计算和原版有一点点不同，wnr的alibi_slope直接加在未scale的S上，参考get_alibi_slope_memref函数
      float qk = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;
      qk *= scale;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }

  // Perform reduction across the threads in the same warp to get the
  // max qk value for each "warp" (not across the thread block yet).
  // The 0-th thread of each thread group already has its max qk value.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();

  // TODO(woosuk): Refactor this part.
  // Get the max qk value for the sequence.
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = __shfl_sync(uint32_t(-1), qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < context_len; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = (HEAD_SIZE + NUM_ROWS_PER_ITER - 1) / NUM_ROWS_PER_ITER;

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

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
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
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

  // Perform reduction within each warp.
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    float acc = accs[i];
#pragma unroll
    for (int mask = NUM_V_VECS_PER_ROW / 2; mask >= 1; mask /= 2) {
      acc += __shfl_xor_sync(uint32_t(-1), acc, mask);
    }
    accs[i] = acc;
  }

  // NOTE(woosuk): A barrier is required because the shared memory space for logits
  // is reused for the output.
  __syncthreads();

  // Perform reduction across warps.
  float* out_smem = reinterpret_cast<float*>(shared_mem);
#pragma unroll
  for (int i = NUM_WARPS; i > 1; i /= 2) {
    int mid = i / 2;
    // Upper warps write to shared memory.
    if (warp_idx >= mid && warp_idx < i) {
      float* dst = &out_smem[(warp_idx - mid) * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          dst[row_idx] = accs[i];
        }
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
        const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
        if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
          accs[i] += src[row_idx];
        }
      }
    }
    __syncthreads();
  }

  // Write the final output.
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(     //
    at::Tensor& out,                                // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                        // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                    // [num_blocks, num_heads_kv, head_size/x, block_size, x]
    const at::Tensor& value_cache,                  // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                                    //
    const at::Tensor& block_tables,                 // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                 // [num_seqs]
    int max_context_len,                            //
    const std::optional<at::Tensor>& alibi_slopes,  // [num_heads]
    cudaStream_t stream                             //
) {
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int num_heads_kv = key_cache.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  auto alibi_slopes_ptr = alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr;
  auto out_ptr = reinterpret_cast<T*>(out.data_ptr());
  auto query_ptr = reinterpret_cast<const T*>(query.data_ptr());
  auto key_cache_ptr = reinterpret_cast<const T*>(key_cache.data_ptr());
  auto value_cache_ptr = reinterpret_cast<const T*>(value_cache.data_ptr());
  auto block_tables_ptr = block_tables.data_ptr<int32_t>();
  auto context_lens_ptr = context_lens.data_ptr<int32_t>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    case 64: {
      constexpr int HEAD_SIZE = 64;
      cudaFuncSetAttribute(single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    case 80: {
      constexpr int HEAD_SIZE = 80;
      cudaFuncSetAttribute(single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    case 96: {
      constexpr int HEAD_SIZE = 96;
      cudaFuncSetAttribute(single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    case 112: {
      constexpr int HEAD_SIZE = 112;
      cudaFuncSetAttribute(single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    case 128: {
      constexpr int HEAD_SIZE = 128;
      cudaFuncSetAttribute(single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    case 256: {
      constexpr int HEAD_SIZE = 256;
      cudaFuncSetAttribute(single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    default:
      throw std::runtime_error("unsupported head size: " + std::to_string(head_size));
      break;
  }
}

void spa(                                          //
    at::Tensor& out,                               // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                       // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                   // [num_blocks, num_heads_kv, head_size/x, block_size, x]
    const at::Tensor& value_cache,                 // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                                   //
    const at::Tensor& block_tables,                // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                // [num_seqs]
    int max_context_len,                           //
    const std::optional<at::Tensor>& alibi_slopes  // [num_heads]
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto odt = out.scalar_type();
  auto qdt = query.scalar_type();
  auto kdt = key_cache.scalar_type();
  auto vdt = value_cache.scalar_type();
  if (odt != qdt || odt != kdt || odt != vdt || odt != at::ScalarType::Half) {
    throw std::runtime_error("output, query, key_cache, and value_cache must have the same dtype of float16");
  }

  using T = half_t;
  int block_size = key_cache.size(3);

  switch (block_size) {
    case 8: {
      single_query_cached_kv_attention_launcher<T, 8>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 16: {
      single_query_cached_kv_attention_launcher<T, 16>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 32: {
      single_query_cached_kv_attention_launcher<T, 32>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    default:
      throw std::runtime_error("unsupported block size: " + std::to_string(block_size));
      break;
  }
}

}  // namespace vllm

namespace tfcc {

template <typename T, int N>
CUTLASS_DEVICE void cr_fma(float (&c)[N], float a, const cutlass::Array<T, N>& b) {
  cutlass::NumericConverter<float, T> cvt;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    c[i] += a * cvt(b[i]);
  }
}

template <typename T, int N>
CUTLASS_DEVICE void cr_arr_to_vec_t(cutlass::Array<T, N>& dst, float (&src)[N]) {
  cutlass::NumericConverter<T, float> cvt;

#pragma unroll
  for (int i = 0; i < N; ++i) {
    dst[i] = cvt(src[i]);
  }
}

struct ReduceOps {
  struct Add {
    __device__ __forceinline__ static float apply(float a, float b) { return a + b; }
    __device__ __forceinline__ static float identity() { return 0.f; }
  };

  struct Max {
    __device__ __forceinline__ static float apply(float a, float b) { return fmaxf(a, b); }
    __device__ __forceinline__ static float identity() { return -FLT_MAX; }
  };
};

struct WarpReduce {
  template <class Op, class T>
  __device__ __forceinline__ static void threadReduce(T& value) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      value = Op::apply(__shfl_xor_sync(uint32_t(-1), value, mask), value);
    }
  }
};

struct CtaReduce {
  template <int NUM_THREADS, class Op, class T>
  __device__ __forceinline__ static void threadReduce(T& value, T* smem) {
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    WarpReduce::threadReduce<Op, T>(value);

    if (lane == 0) {
      smem[warp] = value;
    }

    __syncthreads();

    if (lane < NUM_WARPS) {
      value = smem[lane];
    } else {
      value = Op::identity();
    }

    WarpReduce::threadReduce<Op, T>(value);
  }

  template <int NUM_THREADS, class Op, class T, int N>
  __device__ __forceinline__ static void warpReduce(T (&v)[N], T* smem) {
    constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

    const int tid = threadIdx.x;
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    constexpr int NUM_ELEMENTS_PER_WARP = N * WARP_SIZE;

#pragma unroll
    for (int i = NUM_WARPS; i > 1; i /= 2) {
      int mid = i / 2;
      // Upper warps write to shared memory.
      if (warp >= mid && warp < i) {
        T* dst = &smem[(warp - mid) * NUM_ELEMENTS_PER_WARP];
#pragma unroll
        for (int i = 0; i < N; ++i) {
          const int row_idx = i * WARP_SIZE + lane;
          dst[row_idx] = v[i];
        }
      }
      __syncthreads();

      // Lower warps update the output.
      if (warp < mid) {
        const T* src = &smem[warp * NUM_ELEMENTS_PER_WARP];
#pragma unroll
        for (int i = 0; i < N; ++i) {
          const int row_idx = i * WARP_SIZE + lane;
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

    CtaReduce::threadReduce<NUM_THREADS, ReduceOps::Add, float>(exp_sum, &red_smem[NUM_WARPS]);

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
        // NOTE: We handle this loop independently because __shfl will cause an extra synchronization of threads
        // within
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

    CtaReduce::threadReduce<NUM_THREADS, ReduceOps::Max, float>(qk_max, red_smem);

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
        // NOTE: Experiments have shown that in this case, it will be a little faster to remove the conditional
        // judgment
        float token_logit = token_idx < context_len ? logits[token_idx] : 0.f;
        // float token_logit = logits[token_idx];
        const int offset1 = physical_block_offset * HEAD_SIZE;
        const int offset2 = lane * N;
        vec_t v = *reinterpret_cast<const vec_t*>(v_ptr + offset1 + offset2);
        cr_fma(accs, token_logit, v);
      }
    }
  }

  inline __device__ void storeO() {
    using vec_t = typename Vec<scalar_t, N>::Type;
    __syncthreads();

    CtaReduce::warpReduce<NUM_THREADS, ReduceOps::Add, float, N>(accs, logits);

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

// Grid: (num_heads, num_seqs). One "BLOCK" works on one "SEQ-HEAD".
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(  //
    scalar_t* __restrict__ out,                           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,                       // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,                 // [num_blocks, num_kv_heads, block_size, head_size]
    const scalar_t* __restrict__ v_cache,                 // [num_blocks, num_kv_heads, block_size, head_size]
    const int num_heads_kv,                               //
    const float scale,                                    //
    const int32_t* __restrict__ block_tables,             // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,             // [num_seqs]
    const int max_num_blocks_per_seq,                     //
    const float* __restrict__ alibi_slopes,               // [num_heads]
    const int q_stride,                                   //
    const int kv_block_stride,                            //
    const int kv_head_stride                              //
) {
  static_assert(WARP_SIZE >= BLOCK_SIZE && WARP_SIZE % BLOCK_SIZE == 0);
  static_assert(NUM_THREADS % WARP_SIZE == 0);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / (num_heads / num_heads_kv);
  const int seq_idx = blockIdx.y;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  __shared__ float red_smem[2 * NUM_WARPS];

  using LoadQAndGemmQk = LoadQAndGemmQk_CR1<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
  using Softmax = Softmax<HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
  using GemmPvAndStoreO = GemmPvAndStoreO_CR1<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;

  auto load_q_and_gemm_qk = LoadQAndGemmQk(q, nullptr, k_cache, block_table, logits, red_smem, seq_idx, head_idx,
      kv_head_idx, q_stride, kv_block_stride, kv_head_stride, context_len, num_blocks, scale, alibi_slope);
  load_q_and_gemm_qk.loadQ();
  float qk_max = load_q_and_gemm_qk.gemmQk();

  Softmax::doSoftmax(logits, red_smem, qk_max, context_len);

  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  auto gemm_pv_and_store_o = GemmPvAndStoreO(out_ptr, v_cache, block_table, logits, seq_idx, head_idx, kv_head_idx,
      num_heads, num_blocks, kv_block_stride, kv_head_stride, context_len);
  gemm_pv_and_store_o.gemmPv();
  gemm_pv_and_store_o.storeO();
}

template <typename... Args>
void single_query_cached_kv_attention_launcher(     //
    at::Tensor& out,                                // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                        // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                    // [num_blocks, num_heads_kv, block_size, head_size]
    const at::Tensor& value_cache,                  // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                    //
    const at::Tensor& block_tables,                 // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                 // [num_seqs]
    int max_context_len,                            //
    const std::optional<at::Tensor>& alibi_slopes,  // [num_heads]
    cudaStream_t stream                             //
);

template <int BLOCK_SIZE, int HEAD_SIZE, int NUM_THREADS>
void single_query_cached_kv_attention_launcher(     //
    at::Tensor& out,                                // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                        // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                    // [num_blocks, num_heads_kv, block_size, head_size]
    const at::Tensor& value_cache,                  // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                    //
    const at::Tensor& block_tables,                 // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                 // [num_seqs]
    int max_context_len,                            //
    const std::optional<at::Tensor>& alibi_slopes,  // [num_heads]
    cudaStream_t stream                             //
) {
  using T = half_t;

  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int num_heads_kv = key_cache.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  auto alibi_slopes_ptr = alibi_slopes.has_value() ? alibi_slopes->data_ptr<float>() : nullptr;
  auto out_ptr = reinterpret_cast<T*>(out.data_ptr());
  auto query_ptr = reinterpret_cast<const T*>(query.data_ptr());
  auto key_cache_ptr = reinterpret_cast<const T*>(key_cache.data_ptr());
  auto value_cache_ptr = reinterpret_cast<const T*>(value_cache.data_ptr());
  auto block_tables_ptr = block_tables.data_ptr<int32_t>();
  auto context_lens_ptr = context_lens.data_ptr<int32_t>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  auto kernel = &single_query_cached_kv_attention_kernel<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
  kernel<<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
      scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride,
      kv_head_stride);
}

template <int BLOCK_SIZE, int HEAD_SIZE>
void single_query_cached_kv_attention_launcher(     //
    at::Tensor& out,                                // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                        // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                    // [num_blocks, num_heads_kv, block_size, head_size]
    const at::Tensor& value_cache,                  // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                    //
    const at::Tensor& block_tables,                 // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                 // [num_seqs]
    int max_context_len,                            //
    const std::optional<at::Tensor>& alibi_slopes,  // [num_heads]
    cudaStream_t stream                             //
) {
  // NOTE: The smaller the grid, the larger the blocks should be to ensure sufficient overall concurrency.
  const int num_seqs = query.size(0);
  const int num_heads = query.size(1);
  if (num_seqs * num_heads > 256) {
    constexpr int NUM_THREADS = 128;
    single_query_cached_kv_attention_launcher<BLOCK_SIZE, HEAD_SIZE, NUM_THREADS>(
        out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
  } else {
    constexpr int NUM_THREADS = 256;
    single_query_cached_kv_attention_launcher<BLOCK_SIZE, HEAD_SIZE, NUM_THREADS>(
        out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
  }
}

template <int BLOCK_SIZE>
void single_query_cached_kv_attention_launcher(     //
    at::Tensor& out,                                // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                        // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                    // [num_blocks, num_heads_kv, block_size, head_size]
    const at::Tensor& value_cache,                  // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                    //
    const at::Tensor& block_tables,                 // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                 // [num_seqs]
    int max_context_len,                            //
    const std::optional<at::Tensor>& alibi_slopes,  // [num_heads]
    cudaStream_t stream                             //
) {
  int head_size = query.size(2);

  switch (head_size) {
    case 64: {
      single_query_cached_kv_attention_launcher<BLOCK_SIZE, 64>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 128: {
      single_query_cached_kv_attention_launcher<BLOCK_SIZE, 128>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 256: {
      single_query_cached_kv_attention_launcher<BLOCK_SIZE, 256>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    default:
      throw std::runtime_error("unsupported head size: " + std::to_string(head_size));
      break;
  }
}

void spa(                                          //
    at::Tensor& out,                               // [num_seqs, num_heads, head_size]
    const at::Tensor& query,                       // [num_seqs, num_heads, head_size]
    const at::Tensor& key_cache,                   // [num_blocks, num_heads_kv, block_size, head_size]
    const at::Tensor& value_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                   //
    const at::Tensor& block_tables,                // [num_seqs, max_num_blocks_per_seq]
    const at::Tensor& context_lens,                // [num_seqs]
    int max_context_len,                           //
    const std::optional<at::Tensor>& alibi_slopes  // [num_heads]
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  int block_size = key_cache.size(2);

  switch (block_size) {
    case 8: {
      single_query_cached_kv_attention_launcher<8>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 16: {
      single_query_cached_kv_attention_launcher<16>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 32: {
      single_query_cached_kv_attention_launcher<32>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    default:
      throw std::runtime_error("unsupported block size: " + std::to_string(block_size));
      break;
  }
}

}  // namespace tfcc

#undef WARP_SIZE
#undef MAX
#undef MIN

#include "module.h"
static cr::Register _([](pybind11::module& m) {
  m.def("vllm_spa", &vllm::spa);
  m.def("tfcc_spa", &tfcc::spa);
});
