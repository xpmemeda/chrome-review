#include <algorithm>
#include <cstdint>
#include <iostream>

#include "c10/cuda/CUDAStream.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "torch/all.h"
#include "torch/csrc/api/include/torch/all.h"

#include "../framework/tensor.h"
#include "../module.h"
#include "../support/assert.h"
#include "../support/exceptions.h"
#include "./attentioncomm/attention_dtypes.h"
#include "./attentioncomm/attention_generic.cuh"
#include "./attentioncomm/attention_utils.cuh"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace {

class TmpMemoryHolder {
  char* buffer = nullptr;
  size_t allocated_size = 0;

 public:
  template <class T>
  std::tuple<T*, float*, float*> getTmpBuffers(
      int num_seqs, int num_heads, int max_num_partitions, int head_size, cudaStream_t stream) {
    size_t nbytes_tmp_out = sizeof(T) * num_seqs * num_heads * max_num_partitions * head_size;
    nbytes_tmp_out = DIVIDE_ROUND_UP(nbytes_tmp_out, 64) * 64;
    size_t nbytes_exp_sums = sizeof(float) * num_seqs * num_heads * max_num_partitions;
    nbytes_exp_sums = DIVIDE_ROUND_UP(nbytes_exp_sums, 64) * 64;
    size_t nbytes_max_logits = sizeof(float) * num_seqs * num_heads * max_num_partitions;
    nbytes_max_logits = DIVIDE_ROUND_UP(nbytes_max_logits, 64) * 64;
    size_t nbytes_required = nbytes_tmp_out + nbytes_exp_sums + nbytes_max_logits;

    if (nbytes_required > allocated_size) {
      cr::check_cuda_err(cudaFreeAsync(buffer, stream), "free buffer err");
      cr::check_cuda_err(cudaMallocAsync(&buffer, 2 * nbytes_required, stream), "alloc buffer err.");
      allocated_size = 2 * nbytes_required;
    }

    T* tmp_out_ptr = reinterpret_cast<T*>(buffer);
    float* exp_sums_ptr = reinterpret_cast<float*>(buffer + nbytes_tmp_out);
    float* max_logits_ptr = reinterpret_cast<float*>(buffer + nbytes_tmp_out + nbytes_exp_sums);

    return std::make_tuple(tmp_out_ptr, exp_sums_ptr, max_logits_ptr);
  }
};

static TmpMemoryHolder memory_holder;

using namespace vllm;

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

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE = 0>
__device__ void paged_attention_kernel(        //
    float* __restrict__ exp_sums,              // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,            // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ out,                // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,            // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,      // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,      // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,                    // [num_heads]
    const float scale,                         //
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,          //
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,                        //
    const int kv_block_stride,                 //
    const int kv_head_stride                   //
) {
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  constexpr bool USE_PARTITIONING = PARTITION_SIZE > 0;
  const int context_len = context_lens[seq_idx];
  if (USE_PARTITIONING && partition_idx * PARTITION_SIZE >= context_len) {
    // No work to do. Terminate the thread block.
    return;
  }

  const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
  const int num_blocks_per_partition = USE_PARTITIONING ? PARTITION_SIZE / BLOCK_SIZE : num_context_blocks;

  // [start_block_idx, end_block_idx) is the range of blocks to process.
  const int start_block_idx = USE_PARTITIONING ? partition_idx * num_blocks_per_partition : 0;
  const int end_block_idx = MIN(start_block_idx + num_blocks_per_partition, num_context_blocks);
  const int num_blocks = end_block_idx - start_block_idx;

  // [start_token_idx, end_token_idx) is the range of tokens to process.
  const int start_token_idx = start_block_idx * BLOCK_SIZE;
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, context_len);
  const int num_tokens = end_token_idx - start_token_idx;

  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  constexpr int NUM_THREAD_GROUPS =
      NUM_THREADS / THREAD_GROUP_SIZE;  // Note: This assumes THREAD_GROUP_SIZE divides NUM_THREADS
  assert(NUM_THREADS % THREAD_GROUP_SIZE == 0);
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  // A vector type to store a part of a key or a query.
  // The vector size is configured in such a way that the threads in a thread group
  // fetch or compute 16 bytes at a time.
  // For example, if the size of a thread group is 4 and the data type is half,
  // then the vector size is 16 / (4 * sizeof(half)) == 2.
  constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
  constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;

  const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;
  const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;

  // Load the query to registers.
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
  __syncthreads();  // TODO(naed90): possible speedup if this is replaced with a memory wall right before we use q_vecs

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

  // Iterate over the key blocks.
  // Each warp fetches a block of keys for each iteration.
  // Each thread group in a warp fetches a key from the block, and computes
  // dot product with the query.
  const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by large numbers
    // (e.g., kv_block_stride).
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);

    // Load a key to registers.
    // Each thread in a thread group has a different part of the key.
    // For example, if the the thread group size is 4, then the first thread in the group
    // has 0, 4, 8, ... th vectors of the key, and the second thread has 1, 5, 9, ... th
    // vectors of the key, and so on.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;
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
      // // This includes a reduction across the threads in the same thread group.
      // float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // // Add the ALiBi bias if slopes are given.
      // qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;

      // 这里计算和原版有一点点不同，wnr的alibi_slope直接加在未scale的S上，参考get_alibi_slope_memref函数
      float qk = Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - context_len + 1) : 0;
      qk *= scale;

      if (thread_group_offset == 0) {
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= context_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
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
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
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
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

  // Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();

  // If partitioning is enabled, store the max logit and exp_sum.
  if (USE_PARTITIONING && thread_idx == 0) {
    float* max_logits_ptr =
        max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr =
        exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // Each thread will fetch 16 bytes from the value cache at a time.
  constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
  using V_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using L_vec = typename Vec<scalar_t, V_VEC_SIZE>::Type;
  using Float_L_vec = typename FloatVec<L_vec>::Type;

  constexpr int NUM_V_VECS_PER_ROW = BLOCK_SIZE / V_VEC_SIZE;
  constexpr int NUM_ROWS_PER_ITER = WARP_SIZE / NUM_V_VECS_PER_ROW;
  constexpr int NUM_ROWS_PER_THREAD = DIVIDE_ROUND_UP(HEAD_SIZE, NUM_ROWS_PER_ITER);

  // NOTE(woosuk): We use FP32 for the accumulator for better accuracy.
  float accs[NUM_ROWS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
    accs[i] = 0.f;
  }

  scalar_t zero_value;
  zero(zero_value);
  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {
    // NOTE(woosuk): The block number is stored in int32. However, we cast it to int64
    // because int32 can lead to overflow when this variable is multiplied by large numbers
    // (e.g., kv_block_stride).
    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
    const int physical_block_offset = (lane % NUM_V_VECS_PER_ROW) * V_VEC_SIZE;
    const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
    L_vec logits_vec;
    from_float(logits_vec, *reinterpret_cast<Float_L_vec*>(logits + token_idx - start_token_idx));

    const scalar_t* v_ptr = v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE) {
        const int offset = row_idx * BLOCK_SIZE + physical_block_offset;
        V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + offset);
        if (block_idx == num_context_blocks - 1) {
          // NOTE(woosuk): When v_vec contains the tokens that are out of the context,
          // we should explicitly zero out the values since they may contain NaNs.
          // See https://github.com/vllm-project/vllm/issues/641#issuecomment-1682544472
          scalar_t* v_vec_ptr = reinterpret_cast<scalar_t*>(&v_vec);
#pragma unroll
          for (int j = 0; j < V_VEC_SIZE; j++) {
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
      acc += VLLM_SHFL_XOR_SYNC(acc, mask);
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
    scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                        head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ROWS_PER_THREAD; i++) {
      const int row_idx = lane / NUM_V_VECS_PER_ROW + i * NUM_ROWS_PER_ITER;
      if (row_idx < HEAD_SIZE && lane % NUM_V_VECS_PER_ROW == 0) {
        from_float(*(out_ptr + row_idx), accs[i]);
      }
    }
  }
}

// Grid: (num_heads, num_seqs, max_num_partitions).
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
    int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel_1(float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,                                        // [num_seqs, num_heads, max_num_partitions]
    scalar_t* __restrict__ tmp_out,        // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,        // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,                // [num_heads]
    const float scale,
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride) {
  paged_attention_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>(exp_sums, max_logits, tmp_out, q,
      k_cache, v_cache, num_kv_heads, scale, block_tables, context_lens, max_num_blocks_per_seq, alibi_slopes, q_stride,
      kv_block_stride, kv_head_stride);
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
    int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel_1(scalar_t* __restrict__ out,  // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,        // [num_seqs, num_heads, max_num_partitions]
    const float* __restrict__ max_logits,      // [num_seqs, num_heads, max_num_partitions]
    const scalar_t* __restrict__ tmp_out,      // [num_seqs, num_heads, max_num_partitions, head_size]
    const int32_t* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  if (num_partitions == 1) {
    // No need to reduce. Only copy tmp_out to out.
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
    const scalar_t* tmp_out_ptr =
        tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE + head_idx * max_num_partitions * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += blockDim.x) {
      out_ptr[i] = tmp_out_ptr[i];
    }
    // Terminate the thread block.
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warp_idx = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;

  // Size: 2 * num_partitions.
  extern __shared__ char shared_mem[];
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // Load max logits to shared memory.
  float* shared_max_logits = reinterpret_cast<float*>(shared_mem);
  const float* max_logits_ptr = max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float max_logit = -FLT_MAX;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    const float l = max_logits_ptr[i];
    shared_max_logits[i] = l;
    max_logit = fmaxf(max_logit, l);
  }
  __syncthreads();

  // Get the global max logit.
  // Reduce within the warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = max_logit;
  }
  __syncthreads();
  // Reduce across warps.
  max_logit = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    max_logit = fmaxf(max_logit, VLLM_SHFL_XOR_SYNC(max_logit, mask));
  }
  // Broadcast the max value to all threads.
  max_logit = VLLM_SHFL_SYNC(max_logit, 0);

  // Load rescaled exp sums to shared memory.
  float* shared_exp_sums = reinterpret_cast<float*>(shared_mem + sizeof(float) * num_partitions);
  const float* exp_sums_ptr = exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions;
  float global_exp_sum = 0.0f;
  for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
  }
  __syncthreads();
  global_exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
  const float inv_global_exp_sum = __fdividef(1.0f, global_exp_sum + 1e-6f);

  // Aggregate tmp_out to out.
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE + head_idx * max_num_partitions * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j) {
      acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i]) * shared_exp_sums[j] * inv_global_exp_sum;
    }
    from_float(out_ptr[i], acc);
  }
}

#define LAUNCH_PAGED_ATTENTION_V2(HEAD_SIZE)                                                                          \
  paged_attention_v2_kernel_1<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>                                  \
      <<<grid, block, shared_mem_size, stream>>>(exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr, key_cache_ptr, \
          value_cache_ptr, num_kv_heads, scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,           \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride);                                               \
  paged_attention_v2_reduce_kernel_1<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>                                       \
      <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(                                                       \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, context_lens_ptr, max_num_partitions);

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void paged_attention_v2_launcher(    //
    cr::Tensor& out,                 // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,         // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cr::Tensor& value_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    float scale,                     //
    const cr::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,  // [num_seqs]
    int max_context_len,             //
    const cr::Tensor& alibi_slopes,  // [num_heads]
    cudaStream_t stream              //
) {
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int num_kv_heads = key_cache.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  cr::cr_assert(head_size % thread_group_size == 0, "pagedattention internal error");

  auto alibi_slopes_ptr = alibi_slopes.data<float>();
  auto out_ptr = out.data<T>();
  auto [tmp_out_ptr, exp_sums_ptr, max_logits_ptr] =
      memory_holder.getTmpBuffers<T>(num_seqs, num_heads, max_num_partitions, head_size, stream);
  auto query_ptr = query.data<T>();
  auto key_cache_ptr = key_cache.data<T>();
  auto value_cache_ptr = value_cache.data<T>();
  auto block_tables_ptr = block_tables.data<int32_t>();
  auto context_lens_ptr = context_lens.data<int32_t>();

  dim3 grid(num_heads, num_seqs, max_num_partitions);
  int shared_mem_size = std::max(logits_size, outputs_size);
  dim3 reduce_grid(num_heads, num_seqs);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    case 64:
      LAUNCH_PAGED_ATTENTION_V2(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V2(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V2(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V2(112);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V2(128);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V2(256);
      break;
    default:
      cr::cr_assert(false, "Unsupported head size: " + std::to_string(head_size));
      break;
  }
}

#define CALL_V2_LAUNCHER(T, BLOCK_SIZE)       \
  paged_attention_v2_launcher<T, BLOCK_SIZE>( \
      out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);

#define CALL_V2_LAUNCHER_BLOCK_SIZE(T)                                               \
  switch (block_size) {                                                              \
    case 8:                                                                          \
      CALL_V2_LAUNCHER(T, 8);                                                        \
      break;                                                                         \
    case 16:                                                                         \
      CALL_V2_LAUNCHER(T, 16);                                                       \
      break;                                                                         \
    case 32:                                                                         \
      CALL_V2_LAUNCHER(T, 32);                                                       \
      break;                                                                         \
    default:                                                                         \
      cr::cr_assert(false, "Unsupported block size: " + std::to_string(block_size)); \
      break;                                                                         \
  }

void paged_attention_v2_1_internal(  //
    cr::Tensor& out,                 // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,         // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,     // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cr::Tensor& value_cache,   // [num_blocks, num_kv_heads, head_size, block_size]
    float scale,                     //
    const cr::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,  // [num_seqs]
    int max_context_len,             //
    const cr::Tensor& alibi_slopes,  // [num_heads]
    cudaStream_t stream              //
) {
  int block_size = key_cache.size(3);
  CALL_V2_LAUNCHER_BLOCK_SIZE(uint16_t);
}

void paged_attention_v2_1(                            //
    torch::Tensor& out,                               // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,                       // [num_seqs, num_heads, head_size]
    const torch::Tensor& key_cache,                   // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const torch::Tensor& value_cache,                 // [num_blocks, num_kv_heads, head_size, block_size]
    float scale,                                      //
    const torch::Tensor& block_tables,                // [num_seqs, max_num_blocks_per_seq]
    const torch::Tensor& context_lens,                // [num_seqs]
    int32_t max_context_len,                          //
    const c10::optional<torch::Tensor>& alibi_slopes  // [num_heads]
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  auto cr_out = cr::Tensor::referenceFromTorchTensor(out);
  auto cr_query = cr::Tensor::referenceFromTorchTensor(query);
  auto cr_key_cache = cr::Tensor::referenceFromTorchTensor(key_cache);
  auto cr_value_cache = cr::Tensor::referenceFromTorchTensor(value_cache);
  auto cr_block_tables = cr::Tensor::referenceFromTorchTensor(block_tables);
  auto cr_context_lens = cr::Tensor::referenceFromTorchTensor(context_lens);
  auto cr_alibi_slopes = alibi_slopes ? cr::Tensor::referenceFromTorchTensor(*alibi_slopes) : cr::Tensor();
  paged_attention_v2_1_internal(cr_out, cr_query, cr_key_cache, cr_value_cache, scale, cr_block_tables, cr_context_lens,
      max_context_len, cr_alibi_slopes, stream);
}

static cr::Register _(
    [](pybind11::module& m) { m.def("paged_attention_v2_1", &paged_attention_v2_1, "Layout_K.VLLM; Layout_V.VLLM"); });

}  // namespace
