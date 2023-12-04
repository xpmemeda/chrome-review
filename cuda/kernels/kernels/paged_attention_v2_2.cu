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
#include "../support/exceptions.h"
#include "./attentioncomm/attention_dtypes.h"
#include "./attentioncomm/attention_generic.cuh"
#include "./attentioncomm/attention_utils.cuh"
#include "./attentioncomm/crutils.cuh"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

namespace {

using namespace vllm;

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

// Grid: (num_heads, num_seqs, num_partitions)
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel_2(   //
    float* exp_sums,                           // [num_seqs, num_heads, num_partitions]
    float* max_logits,                         // [num_seqs, num_heads, num_partitions]
    scalar_t* __restrict__ out,                // [num_seqs, num_heads, num_partitions, head_size]
    const scalar_t* __restrict__ q,            // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,      // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const scalar_t* __restrict__ v_cache,      // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_heads_kv,                    //
    const float scale,                         //
    const int32_t* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int32_t* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,          //
    const float* __restrict__ alibi_slopes,    // [num_heads]
    const int q_stride,                        //
    const int kv_block_stride,                 //
    const int kv_head_stride                   //
) {
  static_assert(WARP_SIZE % BLOCK_SIZE == 0 || BLOCK_SIZE % WARP_SIZE == 0);
  static_assert(NUM_THREADS % WARP_SIZE == 0);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  const int kv_head_idx = head_idx / (num_heads / num_heads_kv);
  const int seq_idx = blockIdx.y;
  const float alibi_slope = alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;
  static_assert(PARTITION_SIZE % BLOCK_SIZE == 0);
  const int num_blocks_per_partition = PARTITION_SIZE / BLOCK_SIZE;
  const int32_t* block_table =
      block_tables + seq_idx * max_num_blocks_per_seq + partition_idx * num_blocks_per_partition;
  int context_len = context_lens[seq_idx];
  context_len = MIN(PARTITION_SIZE, context_len - partition_idx * PARTITION_SIZE);
  const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  extern __shared__ char shared_mem[];
  float* logits = reinterpret_cast<float*>(shared_mem);
  __shared__ float red_smem[2 * NUM_WARPS];

  using LoadQAndGemmQk = cr::LoadQAndGemmQk_VLLM1<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
  using Reduce = cr::Reduce;
  using Softmax = cr::Softmax<HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
  using GemmPvAndStoreO = cr::GemmPvAndStoreO_VLLM1<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;

  __shared__ scalar_t q_vecs[HEAD_SIZE];
  auto load_q_and_gemm_qk =
      LoadQAndGemmQk(q, q_vecs, k_cache, block_table, logits, red_smem, seq_idx, head_idx, kv_head_idx, q_stride,
          kv_block_stride, kv_head_stride, context_len, num_blocks, scale, alibi_slope, partition_idx * PARTITION_SIZE);
  load_q_and_gemm_qk.loadQ();
  float qk_max = load_q_and_gemm_qk.gemmQk();

  float exp_sum = Softmax::doSoftmax(logits, red_smem, qk_max, context_len);

  if (threadIdx.x == 0) {
    float* max_logits_ptr =
        max_logits + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions + partition_idx;
    *max_logits_ptr = qk_max;
    float* exp_sums_ptr =
        exp_sums + seq_idx * num_heads * max_num_partitions + head_idx * max_num_partitions + partition_idx;
    *exp_sums_ptr = exp_sum;
  }

  // out = out + partition_idx * HEAD_SIZE;
  scalar_t* out_ptr = out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                      head_idx * max_num_partitions * HEAD_SIZE + partition_idx * HEAD_SIZE;
  auto gemm_pv_and_store_o = GemmPvAndStoreO(out_ptr, v_cache, block_table, logits, seq_idx, head_idx, kv_head_idx,
      num_heads, num_blocks, kv_block_stride, kv_head_stride, context_len);
  gemm_pv_and_store_o.gemmPv();
  __syncthreads();
  // Reduce::reduceToWarp0<Reduce::Add, float, NUM_WARPS, GemmPvAndStoreO::NUM_ROWS_PER_THREAD>(
  //     reinterpret_cast<float*>(shared_mem), gemm_pv_and_store_o.accs);
  gemm_pv_and_store_o.storeO();
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, int HEAD_SIZE, int NUM_THREADS,
    int PARTITION_SIZE>
__global__ void paged_attention_v2_reduce_kernel_2(scalar_t* __restrict__ out,  // [num_seqs, num_heads, head_size]
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
  global_exp_sum = cr::Reduce::reduce<cr::Reduce::Add, float, NUM_WARPS>(&red_smem[NUM_WARPS], global_exp_sum);
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

template <typename T, int BLOCK_SIZE, int NUM_THREADS = 128, int PARTITION_SIZE = 512>
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, head_size/x, block_size, x]
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

  int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  auto [tmp_out_ptr, exp_sums_ptr, max_logits_ptr] =
      memory_holder.getTmpBuffers<T>(num_seqs, num_heads, max_num_partitions, head_size, stream);

  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int logits_size = PARTITION_SIZE * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);
  int reduce_shared_mem_size = 2 * max_num_partitions * sizeof(float);

  dim3 grid(num_heads, num_seqs, max_num_partitions);
  dim3 block(NUM_THREADS);
  dim3 reduce_grid(num_heads, num_seqs);
  switch (head_size) {
    case 64: {
      constexpr int HEAD_SIZE = 64;
      paged_attention_v2_kernel_2<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>
          <<<grid, block, shared_mem_size, stream>>>(exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr,
              key_cache_ptr, value_cache_ptr, num_heads_kv, scale, block_tables_ptr, context_lens_ptr,
              max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride);
      paged_attention_v2_reduce_kernel_2<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>
          <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(
              out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, context_lens_ptr, max_num_partitions);
      break;
    }
    case 128: {
      constexpr int HEAD_SIZE = 128;
      paged_attention_v2_kernel_2<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>
          <<<grid, block, shared_mem_size, stream>>>(exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr,
              key_cache_ptr, value_cache_ptr, num_heads_kv, scale, block_tables_ptr, context_lens_ptr,
              max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride);
      paged_attention_v2_reduce_kernel_2<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>
          <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(
              out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, context_lens_ptr, max_num_partitions);
      break;
    }
    case 256: {
      constexpr int HEAD_SIZE = 256;
      paged_attention_v2_kernel_2<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, PARTITION_SIZE>
          <<<grid, block, shared_mem_size, stream>>>(exp_sums_ptr, max_logits_ptr, tmp_out_ptr, query_ptr,
              key_cache_ptr, value_cache_ptr, num_heads_kv, scale, block_tables_ptr, context_lens_ptr,
              max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride);
      paged_attention_v2_reduce_kernel_2<T, HEAD_SIZE, NUM_THREADS, PARTITION_SIZE>
          <<<reduce_grid, block, reduce_shared_mem_size, stream>>>(
              out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr, context_lens_ptr, max_num_partitions);
      break;
    }
    default:
      throw std::runtime_error("unsupported head size: " + std::to_string(head_size));
      break;
  }
}

void decode_attn(                    //
    cr::Tensor& out,                 // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,         // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,     // [num_blocks, num_heads_kv, head_size/x, block_size, x]
    const cr::Tensor& value_cache,   // [num_blocks, num_heads_kv, head_size, block_size]
    float scale,                     //
    const cr::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,  // [num_seqs]
    int max_context_len,             //
    const cr::Tensor& alibi_slopes,  // [num_heads]
    cudaStream_t stream              //
) {
  using T = uint16_t;
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

#undef WARP_SIZE
#undef MAX
#undef MIN

void decode_attn_2(                                   //
    torch::Tensor& out,                               // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,                       // [num_seqs, num_heads, head_size]
    const torch::Tensor& key_cache,                   // [num_blocks, num_heads_kv, head_size/x, block_size, x]
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

static cr::Register _(
    [](pybind11::module& m) { m.def("paged_attention_v2_2", &decode_attn_2, "Layout_K.VLLM; Layout_V.VLLM"); });

}  // namespace
