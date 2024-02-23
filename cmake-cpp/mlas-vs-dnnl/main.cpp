#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

#include <mlas.h>
#include <omp.h>
#include <dnnl.hpp>

class Duration {
  std::chrono::steady_clock::duration du;

 public:
  Duration() {}
  Duration(std::chrono::steady_clock::duration d) : du(d) {}

  unsigned long microseconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::microseconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long milliseconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::milliseconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long seconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::seconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long minutes() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::minutes>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long hours() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::hours>(du).count();
    return std::max(0lu, cost);
  }
};

class Coster {
  std::chrono::system_clock::time_point start;

 public:
  Coster() : start(std::chrono::system_clock::now()) {}

  void reset() { start = std::chrono::system_clock::now(); }

  Duration lap() const {
    auto now = std::chrono::system_clock::now();
    return Duration(now - start);
  }
};

int dnnl_ncx_convbias2d(int batch, int oc, int ic, int hw, int khw) {
  static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  static dnnl::stream stream(engine);

  dnnl::memory::dims paddings = {khw / 2, khw / 2};
  dnnl::memory::dims strides = {1, 1};
  dnnl::memory::dims dilates = {0, 0};

  dnnl::memory::dims x_dims = {batch, ic, hw, hw};
  dnnl::memory::dims w_dims = {oc, ic, khw, khw};
  dnnl::memory::dims b_dims = {oc};
  dnnl::memory::dims r_dims = {batch, oc, hw, hw};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int w_size = std::accumulate(w_dims.begin(), w_dims.end(), 1, std::multiplies<int>());
  int b_size = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());

  std::vector<float> x(x_size, 1.1f), w(w_size, 1.1f), b(b_size, 1.1f), r(r_size, 0.f);

  auto loop_func = [&]() {
    dnnl::memory::desc x_md = {x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::memory::desc w_md = {w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::memory::desc b_md = {b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::memory::desc r_md = {r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::convolution_forward::desc conv_d(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        x_md, w_md, b_md, r_md, strides, dilates, paddings, paddings);
    dnnl::convolution_forward::primitive_desc conv_pd(conv_d, engine);
    dnnl::memory x_m(
        dnnl::memory::desc{x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw}, engine, x.data());
    dnnl::memory w_m(
        dnnl::memory::desc{w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw}, engine, w.data());
    dnnl::memory b_m(
        dnnl::memory::desc{b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a}, engine, b.data());
    dnnl::memory r_m(
        dnnl::memory::desc{r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw}, engine, r.data());

    auto x_m_required = dnnl::memory(conv_pd.src_desc(), engine);
    if (x_m_required.get_desc() == x_m.get_desc()) {
      x_m_required = x_m;
    } else {
      dnnl::reorder(x_m, x_m_required).execute(stream, x_m, x_m_required);
    }
    auto w_m_required = dnnl::memory(conv_pd.weights_desc(), engine);
    if (w_m_required.get_desc() == w_m.get_desc()) {
      w_m_required = w_m;
    } else {
      dnnl::reorder(w_m, w_m_required).execute(stream, w_m, w_m_required);
    }
    auto b_m_required = dnnl::memory(conv_pd.bias_desc(), engine);
    if (b_m_required.get_desc() == b_m.get_desc()) {
      b_m_required = b_m;
    } else {
      dnnl::reorder(b_m, b_m_required).execute(stream, b_m, b_m_required);
    }
    auto r_m_required = dnnl::memory(conv_pd.dst_desc(), engine);
    if (r_m_required.get_desc() == r_m.get_desc()) {
      r_m_required = r_m;
    }

    dnnl::convolution_forward(conv_pd).execute(
        stream, {{DNNL_ARG_SRC, x_m_required}, {DNNL_ARG_WEIGHTS, w_m_required}, {DNNL_ARG_BIAS, b_m_required},
                    {DNNL_ARG_DST, r_m_required}});
    if (conv_pd.dst_desc() != r_m.get_desc()) {
      dnnl::reorder(r_m_required, r_m).execute(stream, r_m_required, r_m);
    }
    stream.wait();
  };

  loop_func();

  Coster coster;
  for (int i = 0; i < 10; ++i) {
    loop_func();
  }
  return coster.lap().milliseconds();
}

int mlas_ncx_convbias2d(int batch, int ic, int oc, int hw, int khw) {
  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  size_t conv_dimension = 2;
  size_t batch_count = batch;
  size_t group_count = 1;
  size_t in_channels_per_group = ic;
  std::vector<int64_t> input_shape({hw, hw});
  std::vector<int64_t> kernel_shape({khw, khw});
  std::vector<int64_t> dilations({1, 1});
  std::vector<int64_t> paddings({khw / 2, khw / 2, khw / 2, khw / 2});
  std::vector<int64_t> strides({1, 1});
  std::vector<int64_t> output_shape({hw, hw});
  size_t out_channels_per_group = oc;

  std::vector<int> x_dims = {batch, ic, hw, hw};
  std::vector<int> w_dims = {oc, ic, khw, khw};
  std::vector<int> b_dims = {oc};
  std::vector<int> r_dims = {batch, oc, hw, hw};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int w_size = std::accumulate(w_dims.begin(), w_dims.end(), 1, std::multiplies<int>());
  int b_size = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());

  std::vector<float> x(x_size, 1.1f), w(w_size, 1.1f), b(b_size, 1.1f), r(r_size, 0.f);

  auto loop_func = [&]() {
    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize = 0;
    MlasConvPrepare(&Parameters, static_cast<size_t>(conv_dimension), static_cast<size_t>(batch_count),
        static_cast<size_t>(group_count), static_cast<size_t>(in_channels_per_group), input_shape.data(),
        kernel_shape.data(), dilations.data(), paddings.data(), strides.data(), output_shape.data(),
        static_cast<size_t>(out_channels_per_group), &activation, &WorkingBufferSize, 0.0f, nullptr);
    auto working_buffer = std::make_unique<float[]>(WorkingBufferSize);
    MlasConv(&Parameters, x.data(), w.data(), b.data(), working_buffer.get(), r.data(), nullptr);
  };

  loop_func();

  Coster coster;
  for (int i = 0; i < 10; ++i) {
    loop_func();
  }

  return coster.lap().milliseconds();
}

int dnnl_ncx_convbias1d(int batch, int oc, int ic, int hw, int khw) {
  static dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  static dnnl::stream stream(engine);

  dnnl::memory::dims paddings = {khw / 2, khw / 2};
  dnnl::memory::dims strides = {1};
  dnnl::memory::dims dilates = {0};

  dnnl::memory::dims x_dims = {batch, ic, hw};
  dnnl::memory::dims w_dims = {oc, ic, khw};
  dnnl::memory::dims b_dims = {oc};
  dnnl::memory::dims r_dims = {batch, oc, hw};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int w_size = std::accumulate(w_dims.begin(), w_dims.end(), 1, std::multiplies<int>());
  int b_size = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());

  std::vector<float> x(x_size, 1.1f), w(w_size, 1.1f), b(b_size, 1.1f), r(r_size, 0.f);

  auto loop_func = [&]() {
    dnnl::memory::desc x_md = {x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::memory::desc w_md = {w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::memory::desc b_md = {b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::memory::desc r_md = {r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
    dnnl::convolution_forward::desc conv_d(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
        x_md, w_md, b_md, r_md, strides, dilates, paddings, paddings);
    dnnl::convolution_forward::primitive_desc conv_pd(conv_d, engine);
    dnnl::memory x_m(
        dnnl::memory::desc{x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ncw}, engine, x.data());
    dnnl::memory w_m(
        dnnl::memory::desc{w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oiw}, engine, w.data());
    dnnl::memory b_m(
        dnnl::memory::desc{b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a}, engine, b.data());
    dnnl::memory r_m(
        dnnl::memory::desc{r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ncw}, engine, r.data());

    auto x_m_required = dnnl::memory(conv_pd.src_desc(), engine);
    if (x_m_required.get_desc() == x_m.get_desc()) {
      x_m_required = x_m;
    } else {
      dnnl::reorder(x_m, x_m_required).execute(stream, x_m, x_m_required);
    }
    auto w_m_required = dnnl::memory(conv_pd.weights_desc(), engine);
    if (w_m_required.get_desc() == w_m.get_desc()) {
      w_m_required = w_m;
    } else {
      dnnl::reorder(w_m, w_m_required).execute(stream, w_m, w_m_required);
    }
    auto b_m_required = dnnl::memory(conv_pd.bias_desc(), engine);
    if (b_m_required.get_desc() == b_m.get_desc()) {
      b_m_required = b_m;
    } else {
      dnnl::reorder(b_m, b_m_required).execute(stream, b_m, b_m_required);
    }
    auto r_m_required = dnnl::memory(conv_pd.dst_desc(), engine);
    if (r_m_required.get_desc() == r_m.get_desc()) {
      r_m_required = r_m;
    }

    dnnl::convolution_forward(conv_pd).execute(
        stream, {{DNNL_ARG_SRC, x_m_required}, {DNNL_ARG_WEIGHTS, w_m_required}, {DNNL_ARG_BIAS, b_m_required},
                    {DNNL_ARG_DST, r_m_required}});
    if (conv_pd.dst_desc() != r_m.get_desc()) {
      dnnl::reorder(r_m_required, r_m).execute(stream, r_m_required, r_m);
    }
    stream.wait();
  };

  loop_func();

  Coster coster;
  for (int i = 0; i < 10; ++i) {
    loop_func();
  }
  return coster.lap().milliseconds();
}

int mlas_ncx_convbias1d(int batch, int ic, int oc, int hw, int khw) {
  MLAS_ACTIVATION activation;
  activation.ActivationKind = MlasIdentityActivation;

  size_t conv_dimension = 1;
  size_t batch_count = batch;
  size_t group_count = 1;
  size_t in_channels_per_group = ic;
  std::vector<int64_t> input_shape({hw});
  std::vector<int64_t> kernel_shape({khw});
  std::vector<int64_t> dilations({1});
  std::vector<int64_t> paddings({khw / 2, khw / 2});
  std::vector<int64_t> strides({1});
  std::vector<int64_t> output_shape({hw});
  size_t out_channels_per_group = oc;

  std::vector<int> x_dims = {batch, ic, hw};
  std::vector<int> w_dims = {oc, ic, khw};
  std::vector<int> b_dims = {oc};
  std::vector<int> r_dims = {batch, oc, hw};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int w_size = std::accumulate(w_dims.begin(), w_dims.end(), 1, std::multiplies<int>());
  int b_size = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());

  std::vector<float> x(x_size, 1.1f), w(w_size, 1.1f), b(b_size, 1.1f), r(r_size, 0.f);

  auto loop_func = [&]() {
    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize = 0;
    MlasConvPrepare(&Parameters, static_cast<size_t>(conv_dimension), static_cast<size_t>(batch_count),
        static_cast<size_t>(group_count), static_cast<size_t>(in_channels_per_group), input_shape.data(),
        kernel_shape.data(), dilations.data(), paddings.data(), strides.data(), output_shape.data(),
        static_cast<size_t>(out_channels_per_group), &activation, &WorkingBufferSize, 0.0f, nullptr);
    auto working_buffer = std::make_unique<float[]>(WorkingBufferSize);
    MlasConv(&Parameters, x.data(), w.data(), b.data(), working_buffer.get(), r.data(), nullptr);
  };

  loop_func();

  Coster coster;
  for (int i = 0; i < 10; ++i) {
    loop_func();
  }

  return coster.lap().milliseconds();
}

int main() {
  omp_set_num_threads(1);
  std::vector<int> batches({100});
  std::vector<int> iocs({128, 256, 512, 1024});
  std::vector<int> hws({32, 64, 128, 256, 512});
  std::vector<int> khws({1});
  std::ofstream ofs("ncx-2d...txt");
  if (!ofs) {
    return -1;
  }
  char buffer[128];
  for (auto batch : batches) {
    for (auto ioc : iocs) {
      for (auto hw : hws) {
        for (auto khw : khws) {
          int dnnl_cost = dnnl_ncx_convbias1d(batch, ioc, ioc, hw, khw);
          int mlas_cost = mlas_ncx_convbias1d(batch, ioc, ioc, hw, khw);
          sprintf(buffer, "[batch:%d, ioc: %d, hw: %d, khw: %d]\ndnnl vs mlas:%d vs %d\n", batch, ioc, hw, khw,
              dnnl_cost, mlas_cost);
          ofs << buffer;
          ofs.flush();
        }
      }
    }
  }
  return 0;
}