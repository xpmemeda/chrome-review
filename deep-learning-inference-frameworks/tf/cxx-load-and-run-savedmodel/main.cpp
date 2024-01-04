#include <assert.h>
#include <signal.h>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>

#include "rbenchmark_utils.h"
#include "rnpz.h"

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/public/session.h"

tensorflow::DataType npz_to_tensorflow(rnpz::NpyArray::ElementTypeID type_id) {
  switch (type_id) {
    case rnpz::NpyArray::ElementTypeID::DOUBLE:
      return tensorflow::DT_DOUBLE;
    case rnpz::NpyArray::ElementTypeID::FLOAT:
      return tensorflow::DT_FLOAT;
    case rnpz::NpyArray::ElementTypeID::INT64:
      return tensorflow::DT_INT64;
    case rnpz::NpyArray::ElementTypeID::INT32:
      return tensorflow::DT_INT32;
    default:
      throw std::runtime_error("unknown npz element type");
      break;
  }
  return tensorflow::DT_INVALID;
}

tensorflow::Tensor npz_to_tensorflow(const rnpz::NpyArray& npy_array) {
  tensorflow::TensorShape tf_shape;
  for (auto s : npy_array.getShape()) {
    tf_shape.AddDim(s);
  }
  tensorflow::Tensor tf_tensor(npz_to_tensorflow(npy_array.getTypeID()), tf_shape);
  std::memcpy(tf_tensor.data(), npy_array.getData(), npy_array.nbytes());
  return std::move(tf_tensor);
}

std::tuple<std::vector<std::string>, std::vector<std::string>> get_default_input_and_output_names(
    std::shared_ptr<tensorflow::SavedModelBundle> tf_bundle) {
  auto sig_def_it = tf_bundle->meta_graph_def.signature_def().find("serving_default");
  if (sig_def_it == tf_bundle->meta_graph_def.signature_def().end()) {
    // 默认的signature_def名称不是serving_default，则尝试去找不存在init_op和train_op的signature_def
    for (sig_def_it = tf_bundle->meta_graph_def.signature_def().begin();
         sig_def_it != tf_bundle->meta_graph_def.signature_def().end(); ++sig_def_it) {
      if (sig_def_it->first != "__saved_model_init_op" && sig_def_it->first != "__saved_model_train_op") {
        break;
      }
    }
    if (sig_def_it == tf_bundle->meta_graph_def.signature_def().end()) {
      throw std::runtime_error("cannot find default output name");
    }
  }

  const tensorflow::SignatureDef& signature_def = sig_def_it->second;

  std::vector<std::string> input_names;
  for (auto&& input : signature_def.inputs()) {
    input_names.emplace_back(input.second.name());
  }

  std::vector<std::string> output_names;
  for (auto output : signature_def.outputs()) {
    output_names.emplace_back(output.second.name());
  }

  return std::make_tuple(input_names, output_names);
}

std::vector<std::pair<std::string, tensorflow::Tensor>> npz_to_tensorflow(
    const std::string& serving_name_prefix, const rnpz::npz_t& npz_file) {
  std::vector<std::pair<std::string, tensorflow::Tensor>> r;
  for (const auto& [name, npy_array] : npz_file) {
    r.push_back({serving_name_prefix + "_" + name, npz_to_tensorflow(npy_array)});
  }
  return r;
}

int main(int argc, char* argv[]) {
  if (argc != 6) {
    std::cout << "Usage: " << argv[0]
              << " <savedmodel dir> <npz input data> <device: cpu or gpu> <process> <thread per process>" << std::endl;
    return -1;
  }

  std::string model_path = argv[1];
  std::string input_path = argv[2];
  std::string device = argv[3];
  int process_num = std::stoi(argv[4]);
  int thread_num = std::stoi(argv[5]);

  for (int i = 0; i < process_num - 1; ++i) {
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

  auto tf_bundle = std::make_shared<tensorflow::SavedModelBundle>();
  {
    tensorflow::SessionOptions session_options;
    tensorflow::ConfigProto* config = &session_options.config;
    if (device == "cpu") {
      (*config->mutable_device_count())["GPU"] = 0;
    } else if (device == "gpu") {
      (*config->mutable_device_count())["GPU"] = 1;
    } else {
      throw std::runtime_error("invalid device");
    }
    config->set_allow_soft_placement(true);
    config->set_intra_op_parallelism_threads(1);
    config->set_inter_op_parallelism_threads(1);
    config->mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.8 / process_num);

    tensorflow::RunOptions run_options;
    auto s = tensorflow::LoadSavedModel(session_options, run_options, model_path, {"serve"}, tf_bundle.get());
    if (!s.ok()) {
      throw std::runtime_error("cannot load model");
    }
  }

  auto [tf_input_names, tf_output_names] = get_default_input_and_output_names(tf_bundle);
  for (auto&& s : tf_input_names) {
    std::cout << s << std::endl;
  }
  for (auto&& s : tf_output_names) {
    std::cout << s << std::endl;
  }

  auto npz = rnpz::load_npz(input_path);
  auto tf_inputs = npz_to_tensorflow("serving_default", npz);
  std::cout << tf_inputs.size() << std::endl;
  std::vector<tensorflow::Tensor> tf_outputs;
  {
    auto benchmark_func = [&]() {
      tf_outputs.clear();
      auto s = tf_bundle->session->Run(tf_inputs, tf_output_names, {}, &tf_outputs);
      assert(tf_outputs.size() == 1 && "unexpected output size");
      if (!s.ok()) {
        std::cout << s.message() << std::endl;
        throw std::runtime_error("infer error");
      }
    };
    rbenchmark::benchmark_throughput(benchmark_func, thread_num);
  }

  return 0;
}