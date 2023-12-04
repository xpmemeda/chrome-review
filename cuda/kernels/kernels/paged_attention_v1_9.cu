#include <algorithm>
#include <cmath>
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
#include "../support/exceptions.h"
#include "./attentioncomm/attention_dtypes.h"
#include "./attentioncomm/attention_generic.cuh"
#include "./attentioncomm/attention_utils.cuh"
#include "./attentioncomm/crutils.cuh"

#define WARP_SIZE 32
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace {

using namespace vllm;
using namespace cr;

// Grid: (num_heads, num_seqs). One "BLOCK" works on one "SEQ-HEAD".
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE,
    int NUM_THREADS>
__global__ void single_query_cached_kv_attention_kernel_9(  //
    scalar_t* __restrict__ out,                             // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ q,                         // [num_seqs, num_heads, head_size]
    const scalar_t* __restrict__ k_cache,                   // [num_blocks, num_kv_heads, block_size, head_size]
    const scalar_t* __restrict__ v_cache,                   // [num_blocks, num_kv_heads, block_size, head_size]
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

  using LoadQAndGemmQk = LoadQAndGemmQk_CR4<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
  using Softmax = Softmax<HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;
  using GemmPvAndStoreO = GemmPvAndStoreO_CR1<scalar_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;

  constexpr int N = HEAD_SIZE / WARP_SIZE;
  using vec_t = typename Vec<scalar_t, N>::Type;
  __shared__ vec_t smem_k_vecs[LoadQAndGemmQk::NUM_STAGES * NUM_WARPS][WARP_SIZE];
  auto load_q_and_gemm_qk = LoadQAndGemmQk(q, &smem_k_vecs[0][0], k_cache, block_table, logits, red_smem, seq_idx,
      head_idx, kv_head_idx, q_stride, kv_block_stride, kv_head_stride, context_len, num_blocks, scale, alibi_slope);
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
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,               // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                 //
    const cr::Tensor& block_tables,              // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,              // [num_seqs]
    int max_context_len,                         //
    const cr::Tensor& alibi_slopes,              // [num_heads]
    cudaStream_t stream                          //
);

template <int BLOCK_SIZE, int HEAD_SIZE, int NUM_THREADS>
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,               // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                 //
    const cr::Tensor& block_tables,              // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,              // [num_seqs]
    int max_context_len,                         //
    const cr::Tensor& alibi_slopes,              // [num_heads]
    cudaStream_t stream                          //
) {
  using T = uint16_t;

  int num_seqs = block_tables.size(0);
  int num_heads = query.size(1);
  int num_heads_kv = key_cache.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

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

  auto kernel = &single_query_cached_kv_attention_kernel_9<T, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>;

  dim3 grid(num_heads, num_seqs);
  dim3 block(NUM_THREADS);
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
  kernel<<<grid, block, shared_mem_size, stream>>>(out_ptr, query_ptr, key_cache_ptr, value_cache_ptr, num_heads_kv,
      scale, block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq, alibi_slopes_ptr, q_stride, kv_block_stride,
      kv_head_stride);
}

template <int BLOCK_SIZE, int HEAD_SIZE>
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,               // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                 //
    const cr::Tensor& block_tables,              // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,              // [num_seqs]
    int max_context_len,                         //
    const cr::Tensor& alibi_slopes,              // [num_heads]
    cudaStream_t stream                          //
) {
  constexpr int NUM_THREADS = 128;
  single_query_cached_kv_attention_launcher<BLOCK_SIZE, HEAD_SIZE, NUM_THREADS>(
      out, query, key_cache, value_cache, scale, block_tables, context_lens, max_context_len, alibi_slopes, stream);
}

template <int BLOCK_SIZE>
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,               // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                 //
    const cr::Tensor& block_tables,              // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,              // [num_seqs]
    int max_context_len,                         //
    const cr::Tensor& alibi_slopes,              // [num_heads]
    cudaStream_t stream                          //
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

template <>
void single_query_cached_kv_attention_launcher(  //
    cr::Tensor& out,                             // [num_seqs, num_heads, head_size]
    const cr::Tensor& query,                     // [num_seqs, num_heads, head_size]
    const cr::Tensor& key_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
    const cr::Tensor& value_cache,               // [num_blocks, num_heads_kv, block_size, head_size]
    float scale,                                 //
    const cr::Tensor& block_tables,              // [num_seqs, max_num_blocks_per_seq]
    const cr::Tensor& context_lens,              // [num_seqs]
    int max_context_len,                         //
    const cr::Tensor& alibi_slopes,              // [num_heads]
    cudaStream_t stream                          //
) {
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

void decode_attn_9(                                   //
    torch::Tensor& out,                               // [num_seqs, num_heads, head_size]
    const torch::Tensor& query,                       // [num_seqs, num_heads, head_size]
    const torch::Tensor& key_cache,                   // [num_blocks, num_heads_kv, block_size, head_size]
    const torch::Tensor& value_cache,                 // [num_blocks, num_heads_kv, block_size, head_size]
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
    single_query_cached_kv_attention_launcher(cr_out, cr_query, cr_key_cache, cr_value_cache, scale, cr_block_tables,
        cr_context_lens, max_context_len, cr_alibi_slopes, stream);
  } while (0);
  return;
}

static cr::Register _(
    [](pybind11::module& m) { m.def("paged_attention_v1_9", &decode_attn_9, "Layout_K.CR; Layout_V.CR"); });

}  // namespace
