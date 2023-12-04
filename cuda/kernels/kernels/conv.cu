#include "conv.h"

#include <cassert>
#include <iostream>

#include "c10/cuda/CUDAStream.h"
#include "cuda_runtime.h"
#include "cudnn.h"

#include "../support/assert.h"
#include "../support/cudnntraits.h"
#include "../support/exceptions.h"
#include "../support/utils.h"

namespace cr {

namespace {

template <typename T>
void conv3d_v1_kernel(const T* input, const T* filter, T* output,  //
    const int* input_dims, const int* input_strides,               // input
    const int* filter_dims, const int* filter_strides,             // filter
    const int* output_dims, const int* output_strides,             // output
    cudnnHandle_t cudnn_handle,                                    //
    const int* conv_paddings, const int* conv_strides, const int* conv_dilates, int group) {
  cudnnTensorDescriptor_t input_desc;
  check_cuda_err(cudnnCreateTensorDescriptor(&input_desc));
  check_cuda_err(
      cudnnSetTensorNdDescriptor(input_desc, CudnnDataTypeTrait<T>::data_type, 5, input_dims, input_strides));

  cudnnFilterDescriptor_t filter_desc;
  check_cuda_err(cudnnCreateFilterDescriptor(&filter_desc));
  check_cuda_err(
      cudnnSetFilterNdDescriptor(filter_desc, CudnnDataTypeTrait<T>::data_type, CUDNN_TENSOR_NCHW, 5, filter_dims));

  cudnnTensorDescriptor_t output_desc;
  check_cuda_err(cudnnCreateTensorDescriptor(&output_desc));
  check_cuda_err(
      cudnnSetTensorNdDescriptor(output_desc, CudnnDataTypeTrait<T>::data_type, 5, output_dims, output_strides));

  cudnnConvolutionDescriptor_t conv_op_desc;
  check_cuda_err(cudnnCreateConvolutionDescriptor(&conv_op_desc));
  check_cuda_err(cudnnSetConvolutionNdDescriptor(
      conv_op_desc, 3, conv_paddings, conv_strides, conv_dilates, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  check_cuda_err(cudnnSetConvolutionGroupCount(conv_op_desc, group));

  const float alpha = 1.0f, beta = 0.0f;
  cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  do {
    cudnnConvolutionFwdAlgoPerf_t perf;
    check_cuda_err(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn_handle, input_desc, filter_desc, conv_op_desc, output_desc, 1, nullptr, &perf));
    algo = perf.algo;
  } while (0);

  size_t workspace_size = 0;
  void* workspace = nullptr;
  check_cuda_err(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_handle, input_desc, filter_desc, conv_op_desc, output_desc, algo, &workspace_size));
  if (workspace_size > 0) {
    check_cuda_err(cudaMalloc(&workspace, workspace_size));
  }

  check_cuda_err(cudnnConvolutionForward(cudnn_handle, &alpha, input_desc, input, filter_desc, filter, conv_op_desc,
      algo, workspace, workspace_size, &beta, output_desc, output));

  if (workspace_size > 0) {
    check_cuda_err(cudaFree(workspace));
  }

  check_cuda_err(cudnnDestroyTensorDescriptor(input_desc));
  check_cuda_err(cudnnDestroyFilterDescriptor(filter_desc));
  check_cuda_err(cudnnDestroyTensorDescriptor(output_desc));
  check_cuda_err(cudnnDestroyConvolutionDescriptor(conv_op_desc));
}

}  // namespace

void conv_v1(torch::Tensor& input, torch::Tensor& filter, torch::Tensor& output, int group,
    const std::vector<int>& paddings, const std::vector<int>& strides, const std::vector<int>& dilates) {
  std::vector<int> input_dims;
  std::vector<int> input_strides;
  std::vector<int> filter_dims;
  std::vector<int> filter_strides;
  std::vector<int> output_dims;
  std::vector<int> output_strides;
  int rank = input.ndimension();
  for (int i = 0; i < rank; ++i) {
    input_dims.push_back(input.size(i));
    input_strides.push_back(input.stride(i));
    filter_dims.push_back(filter.size(i));
    filter_strides.push_back(filter.stride(i));
    output_dims.push_back(output.size(i));
    output_strides.push_back(output.stride(i));
  }
  int empty_ndim = 5 - rank;
  input_dims.insert(input_dims.begin() + 2, empty_ndim, 1);
  input_strides.insert(input_strides.begin() + 2, empty_ndim, 0);
  filter_dims.insert(filter_dims.begin() + 2, empty_ndim, 1);
  filter_strides.insert(filter_strides.begin() + 2, empty_ndim, 0);
  output_dims.insert(output_dims.begin() + 2, empty_ndim, 1);
  output_strides.insert(output_strides.begin() + 2, empty_ndim, 0);

  std::vector<int> conv_paddings = paddings;
  std::vector<int> conv_strides = strides;
  std::vector<int> conv_dilates = dilates;
  conv_paddings.insert(conv_paddings.begin(), empty_ndim, 0);
  conv_strides.insert(conv_strides.begin(), empty_ndim, 1);
  conv_dilates.insert(conv_dilates.begin(), empty_ndim, 1);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  cudnnHandle_t cudnn_handle;
  check_cuda_err(cudnnCreate(&cudnn_handle));
  check_cuda_err(cudnnSetStream(cudnn_handle, stream));

  print_vec("input_dims: ", input_dims);
  print_vec("input_strides: ", input_strides);
  print_vec("filter_dims: ", filter_dims);
  print_vec("filter_strides: ", filter_strides);
  print_vec("output_dims: ", output_dims);
  print_vec("output_strides: ", output_strides);
  print_vec("conv_paddings: ", conv_paddings);
  print_vec("conv_strides: ", conv_strides);
  print_vec("conv_dilates: ", conv_dilates);
  printf("group: %d\n", group);

  if (input.dtype() == torch::ScalarType::Half) {
    using T = half;
    auto input_ptr = reinterpret_cast<T*>(input.data_ptr());
    auto kernel_ptr = reinterpret_cast<T*>(filter.data_ptr());
    auto output_ptr = reinterpret_cast<T*>(output.data_ptr());
    conv3d_v1_kernel<T>(input_ptr, kernel_ptr, output_ptr, input_dims.data(), input_strides.data(), filter_dims.data(),
        filter_strides.data(), output_dims.data(), output_strides.data(), cudnn_handle, conv_paddings.data(),
        conv_strides.data(), conv_dilates.data(), group);
  } else if (input.dtype() == torch::ScalarType::Float) {
    using T = float;
    auto input_ptr = reinterpret_cast<T*>(input.data_ptr());
    auto kernel_ptr = reinterpret_cast<T*>(filter.data_ptr());
    auto output_ptr = reinterpret_cast<T*>(output.data_ptr());
    conv3d_v1_kernel<T>(input_ptr, kernel_ptr, output_ptr, input_dims.data(), input_strides.data(), filter_dims.data(),
        filter_strides.data(), output_dims.data(), output_strides.data(), cudnn_handle, conv_paddings.data(),
        conv_strides.data(), conv_dilates.data(), group);
  } else {
    cr::cr_assert(false, "invalid dtype");
  }

  check_cuda_err(cudnnDestroy(cudnn_handle));
}

}  // namespace cr
