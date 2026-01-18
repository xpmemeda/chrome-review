// #include "torch/all.h"
// #include "torch/csrc/autograd/python_variable.h"

// #include "./module.h"
// #include "./support/assert.h"

// namespace {

// template <class index_t, int N>
// struct IndexHelper {
//   static_assert(N >= 1);

//   template <class... index_ts>
//   __device__ __host__ __forceinline__ IndexHelper(index_t first_dim, index_ts... other_dims) {
//     static_assert(sizeof...(dims) + 1 == N);
//     index_t dims[N] = {first_dim, static_cast<index_t>(other_dims)...};
//     strides_[N - 1] = 1;
// #pragma unroll
//     for (int i = N - 2; i >= 0; --i) {
//       strides_[i] = strides_[i + 1] * dims[i];
//     }
//   }

//   __device__ __host__ __forceinline__ index_t indicesToOffset(index_t (&indices)[N]) {
//     index_t r = 0;
// #pragma unroll
//     for (int i = 0; i < N; ++i) {
//       r += indices[i] * strides_[i];
//     }
//     return r;
//   }

//   __device__ __host__ __forceinline__ void offsetToIndices(index_t offset, index_t (&indices)[N]) {
//     index_t remainning = offset;
// #pragma unroll
//     for (int i = 0; i < N - 1; ++i) {
//       const index_t idx = remainning / strides_[i];
//       indices[i] = idx;
//       remainning -= idx * strides_[i];
//     }
//     indices[N - 1] = remainning;
//   }

//   index_t strides_[N];
// };

// template <class scalar_t, class index_t, int N>
// struct PermuteKernelParams {
//   template <class scalar_t, class index_t, int N>
//   static PermuteKernelParams get(
//       const index_t* src_dims, const scalar_t* src, const int* permutation, const scalar_t* dst, index_t count) {
//     PermuteKernelParams<scalar_t, index_t, N> params;
//     params.src_index_helper = IndexHelper<index_t, N>(src_dims);
//     index_t dst_dims[N];
//     for (int i = 0; i < N; ++i) {
//       dst_dims[i] = src_dims[permutation[i]];
//     }
//     params.dst_index_helper = IndexHelper<index_t, N>(dst_dims);
//     for (int i = 0; i < N; ++i) {
//       params.permutation[i] = permutation[i];
//     }
//     params.src = src;
//     params.dst = dst;
//     params.count = count;
//     return params;
//   }

//   IndexHelper<index_t, N> src_index_helper;
//   IndexHelper<index_t, N> dst_index_helper;
//   int permutation[N]{};
//   index_t count;
//   const scalar_t* src{};
//   scalar_t* dst{};
// };

// template <class scalar_t, class index_t, int N>
// __global__ permute_kernel_v1(PermuteKernelParams<scalar_t, index_t, N> params) {
//   ;
//   ;
//   ;
// }

// void permute_v1(torch::Tensor& src, torch::Tensor& dst, std::vector<int> permutation) {
//   auto scalar_type = src.scalar_type();
//   if (scalar_type == torch::ScalarType::Float) {
//     using scalar_t = float;
//     using index_t = int64_t;
//     int n = src.ndimension();
//     switch (n) {
// #define SWITCH_CASE(N)                                                                        \
//   case N: {                                                                                   \
//     auto params = PermuteKernelParams::get<scalar_t, index_t, N>(                             \
//         src.sizes().data(), src.data_ptr(), permutation.data(), dst.data_ptr(), src.numel()); \
//     permute_kernel_v1<scalar_t, index_t, N>(params);                                          \
//     break;                                                                                    \
//   }
//       SWITCH_CASE(1);
//       SWITCH_CASE(2);
//       SWITCH_CASE(3);
//       SWITCH_CASE(4);
// #undef SWITCH_CASE
//       default:
//         break;
//     }
//   } else {
//     throw std::runtime_error("not support scalar type.");
//   }
// }  // namespace

// }  // namespace
