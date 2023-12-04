#include <algorithm>
#include <cstdint>
#include <iostream>

#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include "c10/cuda/CUDAStream.h"
#include "torch/all.h"
#include "torch/csrc/api/include/torch/all.h"

#include "../framework/tensor.h"
#include "../module.h"
#include "../support/assert.h"
#include "./attentioncomm/attention_dtypes.h"
#include "./attentioncomm/attention_generic.cuh"
#include "./attentioncomm/attention_utils.cuh"
#include "./attentioncomm/dtype_float16.cuh"

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

}  // namespace vllm

namespace {

using namespace vllm;

// Grid: (num_heads, num_seqs). One "BLOCK" works on one "SEQ-HEAD".
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel(  //
    scalar_t* __restrict__ out,                           // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,                       // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,                 // [num_blocks, num_kv_heads, block_size, head_size]
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

  const int thread_idx = threadIdx.x;
  const int warp_idx = thread_idx / WARP_SIZE;
  const int lane = thread_idx % WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / (num_heads / num_heads_kv);
  const int seq_idx = blockIdx.y;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];

  static_assert(HEAD_SIZE % WARP_SIZE == 0);
  constexpr int VEC_SIZE = HEAD_SIZE / WARP_SIZE;
  using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
  using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;

  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  // NOTE: Opts.
  // 1. 这里尝试所有线程一起来加载Q，而不是只有warp 0来加载，结果是只用warp 0加载更快
  // 2. 这里直接使用寄存器而不是共享内存，结果是使用共享内存更快。
  // 3. 即使不加载Q，耗时不变。
  __shared__ Q_vec q_vecs[WARP_SIZE];
  if (warp_idx == 0) {
    const int vec_idx = lane;
    q_vecs[lane] = *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads();

  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];

  // constexpr int x = 16 / sizeof(scalar_t);
  float qk_max = -FLT_MAX;

  const int32_t* block_table = block_tables + seq_idx * max_num_blocks_per_seq;
  const int context_len = context_lens[seq_idx];
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  for (int token_idx = warp_idx; token_idx < context_len; token_idx += NUM_WARPS) {
    const int block_idx = token_idx / BLOCK_SIZE;
    const int physical_block_number = block_table[block_idx];
    const int physical_block_offset = token_idx % BLOCK_SIZE;

    const scalar_t* k_ptr = k_cache + physical_block_number * kv_block_stride + kv_head_idx * kv_head_stride +
                            physical_block_offset * HEAD_SIZE;
    const int vec_idx = lane;
    Q_vec q_vec = q_vecs[lane];
    K_vec k_vec = *reinterpret_cast<const K_vec*>(k_ptr + vec_idx * VEC_SIZE);
    using A_vec = typename FloatVec<Q_vec>::Type;
    A_vec qk_vec = mul<A_vec, Q_vec, K_vec>(q_vec, k_vec);
    float qk_sum = sum(qk_vec);

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      qk_sum += __shfl_xor_sync(uint32_t(-1), qk_sum, mask);
    }

    if (lane == 0) {
      qk_sum += alibi_slope != 0 ? alibi_slope * (token_idx - context_len + 1) : 0;
      qk_sum *= scale;
      logits[token_idx] = qk_sum;
      qk_max = fmaxf(qk_max, qk_sum);
    }
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

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 1024>
void single_query_cached_kv_attention_launcher_switch_head_size(  //
    cr::Tensor& out,                                              // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                                      // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                                  // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,                                // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                                                  //
    const cr::Tensor& block_tables,                               // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,                               // [num_seqs]
    int max_context_len,                                          //
    const cr::Tensor& alibi_slopes,                               // [num_heads]
    cudaStream_t stream                                           //
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

void single_query_cached_kv_attention_launcher_switch_block_size(  //
    cr::Tensor& out,                                               // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                                       // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                                   // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,                                 // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                                                   //
    const cr::Tensor& block_tables,                                // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,                                // [num_seqs]
    int max_context_len,                                           //
    const cr::Tensor& alibi_slopes,                                // [num_heads]
    cudaStream_t stream                                            //
) {
  using T = uint16_t;
  int block_size = key_cache.size(2);

  switch (block_size) {
    case 8: {
      single_query_cached_kv_attention_launcher_switch_head_size<T, 8>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 16: {
      single_query_cached_kv_attention_launcher_switch_head_size<T, 16>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    case 32: {
      single_query_cached_kv_attention_launcher_switch_head_size<T, 32>(
          out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
      break;
    }
    default:
      throw std::runtime_error("unsupported block size: " + std::to_string(block_size));
      break;
  }
}

void pagedattention(                                  //
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
    single_query_cached_kv_attention_launcher_switch_block_size(cr_out, cr_query, cr_key_cache, cr_value_cache, scale,
        cr_block_tables, cr_context_lens, max_context_len, cr_alibi_slopes, stream);
  } while (0);
  return;
}

static cr::Register _([](pybind11::module& m) { m.def("paged_attention_v1_7", &pagedattention); });

}  // namespace
