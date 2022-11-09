#include "dnnl.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

int main() {
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  // create dims.
  dnnl::memory::dims input_dims = {2, 3, 3, 3};
  dnnl::memory::dims kernel_dims = {3, 3, 3, 3};
  dnnl::memory::dims output_dims = {2, 3, 3, 3};
  dnnl::memory::dims paddings = {1, 1};
  dnnl::memory::dims strides = {1, 1};
  dnnl::memory::dims dilates = {0, 0};
  // create memory desc.
  auto dtype = dnnl::memory::data_type::f32;
  dnnl::memory::desc input_memory_desc(input_dims, dtype,
                                       dnnl::memory::format_tag::nchw);
  dnnl::memory::desc kernel_memory_desc(kernel_dims, dtype,
                                        dnnl::memory::format_tag::oihw);
  dnnl::memory::desc output_memory_desc(output_dims, dtype,
                                        dnnl::memory::format_tag::nchw);
  // create convolution desc.
  dnnl::convolution_forward::desc conv_desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      input_memory_desc, kernel_memory_desc, output_memory_desc, strides,
      dilates, paddings, paddings);

  std::vector<float> input(54, 1.f);
  std::vector<float> kernel(81, 1.f);
  std::vector<float> output(54, 0.f);

  // create memory.
  dnnl::memory input_memory(input_memory_desc, engine, input.data());
  dnnl::memory kernel_memory(kernel_memory_desc, engine, kernel.data());
  dnnl::memory output_memory(output_memory_desc, engine, output.data());

  // forward.
  dnnl::convolution_forward::primitive_desc primitive_desc(conv_desc, engine);
  dnnl::convolution_forward(primitive_desc)
      .execute(stream, {{DNNL_ARG_SRC, input_memory},
                        {DNNL_ARG_WEIGHTS, kernel_memory},
                        {DNNL_ARG_DST, output_memory}});
  stream.wait();

  std::cout << *std::min_element(output.begin(), output.end()) << std::endl;
  std::cout << *std::max_element(output.begin(), output.end()) << std::endl;

  return 0;
}
