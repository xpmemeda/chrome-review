#include <algorithm>
#include <cstdint>
#include <iostream>

#include "c10/cuda/CUDAStream.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "pybind11/pybind11.h"
#include "torch/all.h"
#include "torch/csrc/api/include/torch/all.h"

#include "../framework/tensor.h"
#include "../module.h"
#include "../support/assert.h"
#include "./attentioncomm/attention_dtypes.h"
#include "./attentioncomm/attention_generic.cuh"
#include "./attentioncomm/attention_utils.cuh"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace vllm {

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
__global__ void single_query_cached_kv_attention_kernel_6(  //
    scalar_t* __restrict__ out,                             // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,                         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,                   // [num_blocks, num_kv_heads, block_size, head_size]
    const scalar_t* __restrict__ v_cache,                   // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_heads_kv,                                 //
    const float scale,                                      //
    const int32_t* __restrict__ block_tables,               // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,               // [num_seqs]
    const int max_num_blocks_per_seq,                       //
    const float* __restrict__ alibi_slopes,                 // [num_heads]
    const int q_stride,                                     //
    const int kv_block_stride,                              //
    const int kv_head_stride                                //
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

  // // x == THREAD_GROUP_SIZE * VEC_SIZE
  // // Each thread group fetches x elements from the key at a time.
  // constexpr int x = 16 / sizeof(scalar_t);
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
                                physical_block_offset * HEAD_SIZE;
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;
        const int offset = vec_idx * VEC_SIZE;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset);
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

  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  constexpr int NUM_ELEMENTS_PER_THREAD = HEAD_SIZE / WARP_SIZE;
  constexpr int VEC_SIZE_V = NUM_ELEMENTS_PER_THREAD;
  using LOAD_T = typename Vec<scalar_t, VEC_SIZE_V>::Type;
  static_assert(std::is_same_v<LOAD_T, uint2>);
  constexpr int NUM_LOADS = 1;
  float accs[NUM_ELEMENTS_PER_THREAD];
#pragma unroll
  for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; ++i) {
    accs[i] = 0.f;
  }
  // v_cache [num_blocks, num_heads, block_size, head_size]
  for (int block_idx = warp_idx; block_idx < num_blocks; block_idx += NUM_WARPS) {
    const int physical_block_number = block_table[block_idx];
    const half* v_ptr =
        reinterpret_cast<const half*>(v_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride);
#pragma unroll
    for (int physical_block_offset = 0; physical_block_offset < BLOCK_SIZE; ++physical_block_offset) {
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;
      float token_logit = token_idx < context_len ? logits[token_idx] : 0.f;
      const int offset1 = physical_block_offset * HEAD_SIZE;
#pragma unroll
      for (int load_idx = 0; load_idx < NUM_LOADS; ++load_idx) {
        const int offset2 = (WARP_SIZE * load_idx + lane) * VEC_SIZE_V;
        LOAD_T v = *reinterpret_cast<const LOAD_T*>(v_ptr + offset1 + offset2);
        // __half2float(__ushort_as_half(v.x))
        auto h0 = __half2float(__ushort_as_half(v.x & 0xFFFF));
        auto h1 = __half2float(__ushort_as_half((v.x >> 16) & 0xFFFF));
        auto h2 = __half2float(__ushort_as_half(v.y & 0xFFFF));
        auto h3 = __half2float(__ushort_as_half((v.y >> 16) & 0xFFFF));
        // printf("%u: %f,%f,%f,%f\n", threadIdx.x, h0, h1, h2, h3);
        accs[load_idx * VEC_SIZE_V + 0] += token_logit * h0;
        accs[load_idx * VEC_SIZE_V + 1] += token_logit * h1;
        accs[load_idx * VEC_SIZE_V + 2] += token_logit * h2;
        accs[load_idx * VEC_SIZE_V + 3] += token_logit * h3;
      }
    }
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
      for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; ++i) {
        const int row_idx = i * WARP_SIZE + lane;
        dst[row_idx] = accs[i];
      }
    }
    __syncthreads();

    // Lower warps update the output.
    if (warp_idx < mid) {
      const float* src = &out_smem[warp_idx * HEAD_SIZE];
#pragma unroll
      for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; ++i) {
        const int row_idx = i * WARP_SIZE + lane;
        accs[i] += src[row_idx];
      }
    }
    __syncthreads();
  }

  // Write the final output.
  // out [num_seqs, num_heads, head_size]
  if (warp_idx == 0) {
    scalar_t* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
#pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; ++i) {
      const int row_idx = lane * VEC_SIZE_V + i;
      // const int row_idx = i * WARP_SIZE + lane;
      from_float(*(out_ptr + row_idx), accs[i]);
    }
  }
}

}  // namespace vllm

namespace {

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128>
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,               // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                                 //
    const cr::Tensor& block_tables,              // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,              // [num_seqs]
    int max_context_len,                         //
    const cr::Tensor& alibi_slopes,              // [num_heads]
    cudaStream_t stream                          //
) {
  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int num_heads_kv = key_cache.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  int thread_group_size = MAX(WARP_SIZE / BLOCK_SIZE, 1);
  cr::cr_assert(head_size % thread_group_size == 0, "pagedattention internal error");

  auto alibi_slopes_ptr = alibi_slopes.data<float>();
  auto out_ptr = out.data<T>();
  auto query_ptr = query.data<T>();
  auto key_cache_ptr = key_cache.data<T>();
  auto value_cache_ptr = value_cache.data<T>();
  auto block_tables_ptr = block_tables.data<int32_t>();
  auto context_lens_ptr = context_lens.data<int32_t>();

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_context_len = ((max_context_len + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  int logits_size = padded_max_context_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    // case 64: {
    //   constexpr int HEAD_SIZE = 64;
    //   cudaFuncSetAttribute(vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
    //       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    //   vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
    //       <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
    //       num_heads_kv,
    //           scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
    //           kv_block_stride, kv_head_stride);
    //   break;
    // }
    // case 80: {
    //   constexpr int HEAD_SIZE = 80;
    //   cudaFuncSetAttribute(vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
    //       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    //   vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
    //       <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
    //       num_heads_kv,
    //           scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
    //           kv_block_stride, kv_head_stride);
    //   break;
    // }
    // case 96: {
    //   constexpr int HEAD_SIZE = 96;
    //   cudaFuncSetAttribute(vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
    //       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    //   vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
    //       <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
    //       num_heads_kv,
    //           scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
    //           kv_block_stride, kv_head_stride);
    //   break;
    // }
    // case 112: {
    //   constexpr int HEAD_SIZE = 112;
    //   cudaFuncSetAttribute(vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
    //       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    //   vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
    //       <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
    //       num_heads_kv,
    //           scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
    //           kv_block_stride, kv_head_stride);
    //   break;
    // }
    case 128: {
      constexpr int HEAD_SIZE = 128;
      cudaFuncSetAttribute(vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
          cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
      vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
          <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
              scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
              kv_block_stride, kv_head_stride);
      break;
    }
    // case 256: {
    //   constexpr int HEAD_SIZE = 256;
    //   cudaFuncSetAttribute(vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>,
    //       cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    //   vllm::single_query_cached_kv_attention_kernel_6<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
    //       <<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr,
    //       num_heads_kv,
    //           scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride,
    //           kv_block_stride, kv_head_stride);
    //   break;
    // }
    default:
      throw std::runtime_error("unsupported head size: " + std::to_string(head_size));
      break;
  }
}

void decode_attn(                    //
    cr::Tensor& out,                 // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,         // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,     // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,   // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                     //
    const cr::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,  // [num_seqs]
    int max_context_len,             //
    const cr::Tensor& alibi_slopes,  // [num_heads]
    cudaStream_t stream              //
) {
  using T = uint16_t;
  int block_size = key_cache.size(2);

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

}  // namespace

#undef WARP_SIZE
#undef MAX
#undef MIN

namespace cr {

void decode_attn_6(                                   //
    torch::Tensor& out,                               // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,                       // [num_seqs, num_heads, head_size]
    const torch::Tensor& key_cache,                   // [num_blocks, num_heads_kv, block_size, head_size]
    const torch::Tensor& value_cache,                 // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                                      //
    const torch::Tensor& block_tables,                // [num_seqs, max_num_blocks_per_seq]
    const torch::Tensor& context_lens,                // [num_seqs]
    int max_context_len,                              //
    const c10::optional<torch::Tensor>& alibi_slopes  // [num_heads]
) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  do {
    auto cr_out = cr::Tensor::referenceFromTorchTensor(out);
    auto cr_query = cr::Tensor::referenceFromTorchTensor(query);
    auto cr_key_cache = cr::Tensor::referenceFromTorchTensor(key_cache);
    auto cr_value_cache = cr::Tensor::referenceFromTorchTensor(value_cache);
    auto cr_block_tables = cr::Tensor::referenceFromTorchTensor(block_tables);
    auto cr_context_lens = cr::Tensor::referenceFromTorchTensor(context_lens);
    auto cr_alibi_slopes = alibi_slopes ? cr::Tensor::referenceFromTorchTensor(*alibi_slopes) : cr::Tensor();
    decode_attn(cr_out, cr_query, cr_key_cache, cr_value_cache, scale, cr_block_tables, cr_context_lens,
        max_context_len, cr_alibi_slopes, stream);
  } while (0);
  return;
}

namespace {

static Register _([](pybind11::module& m) { m.def("paged_attention_v1_6", &decode_attn_6); });

}  // namespace

}  // namespace cr
