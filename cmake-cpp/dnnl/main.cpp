#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <omp.h>
#include <dnnl.hpp>

template <typename T>
void print_vector(const std::vector<T>& vec) {
  for (auto i : vec) std::cout << i << "; ";
  std::cout << std::endl;
}

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

int matmul() {
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  Coster coster;
  dnnl::memory::dims x_dims = {2, 700, 800}, y_dims = {2, 800, 900}, r_dims = {2, 700, 900};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int y_size = std::accumulate(y_dims.begin(), y_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());

  dnnl::memory::desc x_md = {x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc y_md = {y_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc r_md = {r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};

  dnnl::matmul::desc matmul_d(x_md, y_md, r_md);
  dnnl::matmul::primitive_desc matmul_pd(matmul_d, engine);

  std::vector<float> x(x_size, 1.1f), y(y_size, 1.1f), r(r_size, 0.f);

  std::cout << "init: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  dnnl::memory x_m(
      dnnl::memory::desc{x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, engine, x.data());
  dnnl::memory y_m(
      dnnl::memory::desc{y_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::abc}, engine, y.data());

  std::cout << "set-memory: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  auto x_m_required = dnnl::memory(matmul_pd.src_desc(), engine);
  if (x_m_required.get_desc() == x_m.get_desc()) {
    x_m_required = x_m;
  } else {
    dnnl::reorder(x_m, x_m_required).execute(stream, x_m, x_m_required);
  }
  auto y_m_required = dnnl::memory(matmul_pd.weights_desc(), engine);
  if (y_m_required.get_desc() == y_m.get_desc()) {
    y_m_required = y_m;
  } else {
    dnnl::reorder(y_m, y_m_required).execute(stream, y_m, y_m_required);
  }
  auto r_m_required = dnnl::memory(matmul_pd.dst_desc(), engine, r.data());

  std::cout << "reorder: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  stream.wait();
  for (int i = 0; i < 100; ++i) {
    dnnl::matmul(matmul_pd).execute(
        stream, {{DNNL_ARG_SRC, x_m_required}, {DNNL_ARG_WEIGHTS, y_m_required}, {DNNL_ARG_DST, r_m_required}});
  }
  stream.wait();

  std::cout << "execute: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  return 0;
}

int conv2d() {
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  Coster coster;
  dnnl::memory::dims x_dims = {2, 8, 800, 800};
  dnnl::memory::dims w_dims = {8, 8, 3, 3};
  dnnl::memory::dims r_dims = {2, 8, 800, 800};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int w_size = std::accumulate(w_dims.begin(), w_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());
  dnnl::memory::dims paddings = {1, 1};
  dnnl::memory::dims strides = {1, 1};
  dnnl::memory::dims dilates = {0, 0};

  dnnl::memory::desc x_md = {x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc w_md = {w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc r_md = {r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::convolution_forward::desc conv_d(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, x_md,
      w_md, r_md, strides, dilates, paddings, paddings);
  dnnl::convolution_forward::primitive_desc conv_pd(conv_d, engine);

  std::cout << "init-descriptions: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  std::vector<float> x(x_size, 1.1f), w(w_size, 1.1f), r(r_size, 0.f);
  std::cout << "init: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  dnnl::memory x_m(
      dnnl::memory::desc{x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw}, engine, x.data());
  dnnl::memory w_m(
      dnnl::memory::desc{w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oihw}, engine, w.data());
  dnnl::memory r_m(
      dnnl::memory::desc{r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::nchw}, engine, r.data());

  std::cout << "set-memory: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

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
  auto r_m_required = dnnl::memory(conv_pd.dst_desc(), engine);
  if (r_m_required.get_desc() == r_m.get_desc()) {
    r_m_required = r_m;
  }
  stream.wait();

  std::cout << "reorder: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  for (int i = 0; i < 100; ++i) {
    dnnl::convolution_forward(conv_pd).execute(
        stream, {{DNNL_ARG_SRC, x_m_required}, {DNNL_ARG_WEIGHTS, w_m_required}, {DNNL_ARG_DST, r_m_required}});
  }
  stream.wait();

  std::cout << "execute: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  return 0;
}

int conv3d_bias() {
  dnnl::engine engine(dnnl::engine::kind::cpu, 0);
  dnnl::stream stream(engine);

  Coster coster;
  dnnl::memory::dims x_dims = {2, 8, 112, 112, 112};
  dnnl::memory::dims w_dims = {8, 8, 3, 3, 3};
  dnnl::memory::dims r_dims = {2, 8, 112, 112, 112};
  dnnl::memory::dims b_dims = {8};
  int x_size = std::accumulate(x_dims.begin(), x_dims.end(), 1, std::multiplies<int>());
  int w_size = std::accumulate(w_dims.begin(), w_dims.end(), 1, std::multiplies<int>());
  int r_size = std::accumulate(r_dims.begin(), r_dims.end(), 1, std::multiplies<int>());
  int b_size = std::accumulate(b_dims.begin(), b_dims.end(), 1, std::multiplies<int>());
  dnnl::memory::dims paddings = {1, 1, 1};
  dnnl::memory::dims strides = {1, 1, 1};
  dnnl::memory::dims dilates = {0, 0, 0};

  dnnl::memory::desc x_md = {x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc w_md = {w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc r_md = {r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::memory::desc b_md = {b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any};
  dnnl::convolution_forward::desc conv_d(dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, x_md,
      w_md, b_md, r_md, strides, dilates, paddings, paddings);
  dnnl::convolution_forward::primitive_desc conv_pd(conv_d, engine);

  std::cout << "init-descriptions: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  std::vector<float> x(x_size, 0.f), w(w_size, 0.f), r(r_size, 0.f);
  std::vector<float> b({0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f});
  std::cout << "init: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  dnnl::memory x_m(
      dnnl::memory::desc{x_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ncdhw}, engine, x.data());
  dnnl::memory w_m(
      dnnl::memory::desc{w_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::oidhw}, engine, w.data());
  dnnl::memory r_m(
      dnnl::memory::desc{r_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ncdhw}, engine, r.data());
  dnnl::memory b_m(
      dnnl::memory::desc{b_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::a}, engine, b.data());

  std::cout << "set-memory: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

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
  stream.wait();

  std::cout << "reorder: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  for (int i = 0; i < 10; ++i) {
    dnnl::convolution_forward(conv_pd).execute(
        stream, {{DNNL_ARG_SRC, x_m_required}, {DNNL_ARG_WEIGHTS, w_m_required}, {DNNL_ARG_BIAS, b_m_required},
                    {DNNL_ARG_DST, r_m_required}});
    if (conv_pd.dst_desc() != r_m.get_desc()) {
      dnnl::reorder(r_m_required, r_m).execute(stream, r_m_required, r_m);
    }
  }
  stream.wait();

  std::cout << "execute: " << coster.lap().milliseconds() << "(ms)" << std::endl;
  coster.reset();

  return 0;
}

int main(int argc, char* argv[]) {
  int thread_num = argc < 2 ? 1 : std::stoi(argv[1]);
  if (thread_num > omp_get_max_threads()) throw std::runtime_error("use too many threads");
  omp_set_num_threads(thread_num);
  std::cout << "========== matmul ===========" << std::endl;
  matmul();
  std::cout << "========== conv2d ===========" << std::endl;
  conv2d();
  std::cout << "========== conv3d ===========" << std::endl;
  conv3d_bias();
  return 0;
}
