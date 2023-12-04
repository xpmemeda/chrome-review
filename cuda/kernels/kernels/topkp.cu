#include <cstdint>

#include "c10/cuda/CUDAStream.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "torch/all.h"

#include "../module.h"
#include "../support/assert.h"

namespace {

void topkp_1(torch::Tensor& probs, torch::Tensor& logits, torch::Tensor& ks, torch::Tensor& ps) {
  auto [logits_sort, logits_idx] = logits.sort(-1, /*descending*/ false);

  // Apply top-k.
  torch::Tensor top_k_mask = logits_sort.size(1) - ks;
  // Get all the top_k values.
  top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(1));
  top_k_mask = logits_sort < top_k_mask;
  logits_sort.masked_fill_(top_k_mask, -std::numeric_limits<float>::infinity());

  // Apply top-p.
  torch::Tensor probs_sort = logits_sort.softmax(-1);
  torch::Tensor probs_sum = probs_sort.cumsum(-1);
  torch::Tensor top_p_mask = probs_sum <= (1 - ps.unsqueeze(1));
  // at least one, top_p_mask[:, -1] = False
  top_p_mask.index_fill_(1, torch::tensor({-1}, torch::TensorOptions().device(logits.device())), false);
  logits_sort.masked_fill_(top_p_mask, -std::numeric_limits<float>::infinity());

  // Re - sort the probabilities.
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(logits_idx.device());
  torch::Tensor src = torch::arange(logits_idx.size(-1), options).expand_as(logits_idx);
  torch::Tensor logits_idx_inv = torch::empty(logits_idx.sizes(), options).scatter_(-1, logits_idx, src);
  probs.copy_(torch::gather(logits_sort, -1, logits_idx_inv).softmax(-1));
}

static cr::Register _([](pybind11::module& m) { m.def("topkp_1", &topkp_1); });

}  // namespace
