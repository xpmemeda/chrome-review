#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include "unistd.h"

#include <onnxruntime_cxx_api.h>
#include <rnpz.h>

class Duration {
  std::chrono::steady_clock::duration du;

 public:
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

ONNXTensorElementDataType npz_to_onnx(rnpz::NpyArray::ElementTypeID type_id) {
  switch (type_id) {
    case rnpz::NpyArray::ElementTypeID::DOUBLE:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case rnpz::NpyArray::ElementTypeID::FLOAT:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case rnpz::NpyArray::ElementTypeID::INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case rnpz::NpyArray::ElementTypeID::INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    default:
      throw std::runtime_error("unknown npz element type");
      break;
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

rnpz::NpyArray::ElementTypeID onnx_to_npz(ONNXTensorElementDataType ort_dtype) {
  switch (ort_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return rnpz::NpyArray::ElementTypeID::DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return rnpz::NpyArray::ElementTypeID::FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return rnpz::NpyArray::ElementTypeID::INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return rnpz::NpyArray::ElementTypeID::INT32;
    default:
      throw std::runtime_error("unknown ort dtype");
      break;
  }
  return rnpz::NpyArray::ElementTypeID::UNKNOW;
}

Ort::Value npz_to_onnx(const rnpz::NpyArray& npy_array) {
  Ort::MemoryInfo mem_info =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
  std::vector<int64_t> shape;
  for (size_t i = 0; i < npy_array.getShape().size(); ++i) {
    shape.push_back(npy_array.getShape()[i]);
  }
  return Ort::Value::CreateTensor(mem_info, const_cast<void*>(npy_array.getData()), npy_array.nbytes(), shape.data(),
      shape.size(), npz_to_onnx(npy_array.getTypeID()));
}

rnpz::NpyArray onnx_to_npz(const Ort::Value& ort_value) {
  auto npz_element_type = onnx_to_npz(ort_value.GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementType());
  std::vector<size_t> shape;
  for (int64_t s : ort_value.GetTypeInfo().GetTensorTypeAndShapeInfo().GetShape()) {
    shape.push_back(static_cast<size_t>(s));
  }
  size_t numel = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
  size_t nbytes = numel * rnpz::NpyArray::getElementByteWidth(npz_element_type);
  auto buffer = std::make_unique<char[]>(nbytes);
  std::memcpy(buffer.get(), ort_value.GetTensorData<void*>(), nbytes);
  return rnpz::NpyArray(shape, npz_element_type, std::move(buffer));
}

std::vector<Ort::Value> npz_to_onnx(const std::vector<std::string>& input_names, const rnpz::npz_t& npz_file) {
  std::vector<Ort::Value> r;
  for (auto&& name : input_names) {
    for (auto it = npz_file.begin(); it != npz_file.end(); ++it) {
      if (it->first == name) {
        r.push_back(npz_to_onnx(it->second));
      }
    }
  }
  if (r.size() != input_names.size()) {
    throw std::runtime_error("invalid input");
  }
  return r;
}

rnpz::npz_t onnx_to_npz(const std::vector<std::string>& names, const std::vector<Ort::Value>& ort_values) {
  if (names.size() != ort_values.size()) {
    throw std::runtime_error("...");
  }
  rnpz::npz_t r;
  for (size_t i = 0; i < names.size(); ++i) {
    r.insert({names[i], onnx_to_npz(ort_values[i])});
  }
  return r;
}

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size(); ++i) {
    ss << v[i] << 'x';
  }
  return ss.str().substr(0, ss.str().size() - 1);
}

int calculate_product(const std::vector<std::int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

void init_cpu_options(Ort::SessionOptions* session_options) {
  session_options->SetIntraOpNumThreads(1);
  session_options->SetInterOpNumThreads(1);
}

void init_gpu_options(Ort::SessionOptions* session_options) {
  OrtCUDAProviderOptions cuda_provider_options;
  cuda_provider_options.device_id = 0;
  cuda_provider_options.do_copy_in_default_stream = 1;
  session_options->AppendExecutionProvider_CUDA(cuda_provider_options);
}

void process_function(const std::string& model_path, const std::string& npz_inputs, const std::string& device,
    std::atomic<size_t>* process_cnt, std::atomic<size_t>* total_cost, std::atomic<size_t>* max_cost) {
  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort-demo");
  Ort::SessionOptions session_options;
  if (device == "cpu") {
    init_cpu_options(&session_options);
  } else if (device == "gpu") {
    init_gpu_options(&session_options);
  } else {
    throw std::runtime_error("unknown device: " + device);
  }
  Ort::Session session = Ort::Session(env, model_path.c_str(), session_options);

  // print name/shape of inputs
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::string> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetInputCount(); i++) {
    input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
    std::vector<int64_t> input_shape = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shape) << std::endl;
    input_shapes.push_back(input_shape);
  }

  // print name/shape of outputs
  std::vector<std::string> output_names;
  std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
  for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
    output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
    auto output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
  }

  std::vector<const char*> input_names_char(input_names.size(), nullptr);
  std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
      [&](const std::string& str) { return str.c_str(); });

  std::vector<const char*> output_names_char(output_names.size(), nullptr);
  std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
      [&](const std::string& str) { return str.c_str(); });

  auto npz_file = rnpz::load_npz(npz_inputs);

  while (true) {
    Coster coster;
    auto input_tensors = npz_to_onnx(input_names, npz_file);
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names_char.data(), input_tensors.data(),
        input_names_char.size(), output_names_char.data(), output_names_char.size());
    rnpz::npz_t npz_outputs = onnx_to_npz(output_names, output_tensors);
    size_t current_cost = coster.lap().microseconds();
    process_cnt->operator+=(1);
    total_cost->operator+=(current_cost);
    if (current_cost > max_cost->load()) {
      max_cost->store(current_cost);
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0] << " <onnx model> <npz inputs> <device: cpu, gpu> <process cnt> <thread cnt>"
              << std::endl;
    return -1;
  }
  std::string model_path = argv[1];
  std::string npz_inputs = argv[2];
  std::string device = argv[3];
  int process_count = std::stoi(argv[4]);
  int thread_count = std::stoi(argv[5]);

  int worker_id = 0;
  for (; worker_id < process_count - 1; ++worker_id) {
    int pid = fork();
    if (pid < 0) {
      std::cout << "fork err" << std::endl;
      kill(0, SIGKILL);
      abort();
    }
    if (pid == 0) {
      break;
    }
  }

  std::atomic<size_t> process_cnt(0);  // 处理的请求数
  std::atomic<size_t> total_cost(0);   // 总的耗时
  std::atomic<size_t> max_cost(0);     // 最大耗时

  for (int i = 0; i < thread_count; ++i) {
    new std::thread(process_function, model_path, npz_inputs, device, &process_cnt, &total_cost, &max_cost);
  }

  while (true) {
    size_t old_cnt = process_cnt.load();
    size_t old_cost = total_cost.load();
    max_cost.store(0);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    size_t new_cnt = process_cnt.load();
    size_t new_cost = total_cost.load();

    double qps = static_cast<double>(new_cnt - old_cnt) / 10.;
    double avgcost = static_cast<double>(new_cost - old_cost) / static_cast<double>(new_cnt - old_cnt) / 1000;
    double maxcost = static_cast<double>(max_cost.load()) / 1000;

    printf("qps: %f, avgcost: %f, maxcost: %f\n", qps, avgcost, maxcost);
  }

  return 0;
}