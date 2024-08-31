#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "chrome-review/cr.h"

using namespace nvinfer1;
using namespace nvonnxparser;

class Printer {
 public:
  virtual void print() const = 0;
  virtual ~Printer() {}
};

class TensorPrinter : public Printer {
 public:
  enum class DataType : int32_t { kFLOAT = 0, kHALF = 1, kINT8 = 2, kINT32 = 3, kBOOL = 4, kUINT8 = 5, kFP8 = 6 };

  TensorPrinter(const std::string& name, const std::vector<int>& shape, DataType data_type)
      : name_(name), shape_(shape), data_type_(data_type) {}
  void print() const override {
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < shape_.size(); ++i) {
      ss << shape_[i] << ", ";
    }
    std::string shape_str = ss.str().substr(0, ss.str().size() - 2);
    shape_str.append(")");

    std::cout << std::left << std::setw(20) << name_ << std::left << std::setw(20) << shape_str
              << getTypeStr(data_type_) << std::endl;
  }

 private:
  const char* getTypeStr(DataType data_type) const {
    switch (data_type) {
      case DataType::kFLOAT:
        return "FLOAT";
      case DataType::kHALF:
        return "HALF";
      case DataType::kINT8:
        return "INT8";
      case DataType::kINT32:
        return "INT32";
      case DataType::kBOOL:
        return "BOOL";
      case DataType::kUINT8:
        return "UINT8";
      case DataType::kFP8:
        return "FP8";
      default:
        throw std::runtime_error("unrecognized data type");
    }
    return "";
  }

 private:
  std::string name_;
  std::vector<int> shape_;
  DataType data_type_;
};

TensorPrinter::DataType convert_trt(nvinfer1::DataType data_type) {
  using DataType = nvinfer1::DataType;
  switch (data_type) {
    case DataType::kFLOAT:
      return TensorPrinter::DataType::kFLOAT;
    case DataType::kHALF:
      return TensorPrinter::DataType::kHALF;
    case DataType::kINT8:
      return TensorPrinter::DataType::kINT8;
    case DataType::kINT32:
      return TensorPrinter::DataType::kINT32;
    case DataType::kBOOL:
      return TensorPrinter::DataType::kBOOL;
    case DataType::kUINT8:
      return TensorPrinter::DataType::kUINT8;
    case DataType::kFP8:
      return TensorPrinter::DataType::kFP8;
    default:
      throw std::runtime_error("unrecognized data type");
  }
  return TensorPrinter::DataType();
}

void print_input_and_output_tensor_info(nvinfer1::INetworkDefinition* network) {
  std::cout << "(ins)" << std::endl;
  for (int i = 0; i < network->getNbInputs(); ++i) {
    std::string input_name = network->getInput(i)->getName();
    auto dims = network->getInput(i)->getDimensions();
    std::vector<int> input_shape;
    for (int j = 0; j < dims.nbDims; ++j) {
      input_shape.push_back(dims.d[j]);
    }
    TensorPrinter(input_name, input_shape, convert_trt(network->getInput(i)->getType())).print();
  }
  std::cout << "(outs)" << std::endl;
  for (int i = 0; i < network->getNbOutputs(); ++i) {
    std::string output_name = network->getOutput(i)->getName();
    auto dims = network->getOutput(i)->getDimensions();
    std::vector<int> output_shape;
    for (int j = 0; j < dims.nbDims; ++j) {
      output_shape.push_back(dims.d[j]);
    }
    TensorPrinter(output_name, output_shape, convert_trt(network->getOutput(i)->getType())).print();
  }
}

class TrtIOBuffer {
 public:
  TrtIOBuffer(nvinfer1::ICudaEngine* engine, const cr::npz::npz_t& npz) {
    for (int i = 0; i < engine->getNbBindings(); ++i) {
      size_t count = 10 * 256 * sizeof(float);
      cuda_buffer_.push_back(nullptr);
      cudaError_t ret = cudaMalloc(&cuda_buffer_.back(), count);
      if (ret != cudaSuccess) {
        throw std::runtime_error("cannot alloc cuda memory");
      }

      // auto input_name = engine->getBindingName(i);
      // auto it = npz.find(input_name);
      // if (it == npz.end()) {
      //   throw std::runtime_error("cannot find input in npz");
      // }
      // auto& ndarray = it->second;
      // cuda_buffer_.push_back(nullptr);
      // auto ret = cudaMalloc(&cuda_buffer_.back(), ndarray.nbytes());
      // if (ret != cudaSuccess) {
      //   throw std::runtime_error("cannot alloc cuda memory");
      // }
      // cudaMemcpy(cuda_buffer_.back(), ndarray.getData(), ndarray.nbytes(), cudaMemcpyHostToDevice);
    };
  }

  void** getBuffer() { return cuda_buffer_.data(); }

 private:
  std::vector<void*> cuda_buffer_;
};

class Logger : public ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity == Severity::kERROR) {
      std::cerr << msg << std::endl;
    }
  }
};

std::vector<char> readFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (file.read(buffer.data(), size)) {
    return buffer;
  } else {
    throw std::runtime_error("Failed to read file: " + filename);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <onnx model path> <npz data path>\n", argv[0]);
  }

  std::string onnx_model_path = argv[1];
  std::string npz_data_path = argv[2];

  // Initialize TensorRT logger
  Logger logger;

  // Read ONNX model file
  std::string onnxFile = onnx_model_path;
  auto onnxModel = readFile(onnxFile);

  // Create TensorRT builder, network, and parser
  auto builder = std::unique_ptr<IBuilder>(createInferBuilder(logger));
  const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  auto parser = std::unique_ptr<IParser>(createParser(*network, logger));

  // Parse ONNX model
  if (!parser->parse(onnxModel.data(), onnxModel.size())) {
    std::cerr << "Failed to parse ONNX model" << std::endl;
    return -1;
  }

  print_input_and_output_tensor_info(network.get());

  auto profile = builder->createOptimizationProfile();
  int batch_size = 1;
  profile->setDimensions("src", OptProfileSelector::kMIN, nvinfer1::Dims2(batch_size, 256));
  profile->setDimensions("src", OptProfileSelector::kOPT, nvinfer1::Dims2(batch_size, 256));
  profile->setDimensions("src", OptProfileSelector::kMAX, nvinfer1::Dims2(batch_size, 256));
  profile->setDimensions("tgt", OptProfileSelector::kMIN, nvinfer1::Dims2(batch_size, 98));
  profile->setDimensions("tgt", OptProfileSelector::kOPT, nvinfer1::Dims2(batch_size, 98));
  profile->setDimensions("tgt", OptProfileSelector::kMAX, nvinfer1::Dims2(batch_size, 98));
  profile->setDimensions("seg", OptProfileSelector::kMIN, nvinfer1::Dims2(batch_size, 256));
  profile->setDimensions("seg", OptProfileSelector::kOPT, nvinfer1::Dims2(batch_size, 256));
  profile->setDimensions("seg", OptProfileSelector::kMAX, nvinfer1::Dims2(batch_size, 256));

  auto config = std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
  config->setMaxWorkspaceSize(1 << 30);
  config->addOptimizationProfile(profile);

  auto engine = std::unique_ptr<ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
  std::cout << engine->getNbBindings() << std::endl;

  // Create execution context
  auto context = std::unique_ptr<IExecutionContext>(engine->createExecutionContext());

  // Allocate memory for input and output
  auto npz = cr::npz::load_npz(npz_data_path);
  auto buffer = TrtIOBuffer(engine.get(), npz);

  // Execute inference
  void** bu = buffer.getBuffer();

  auto fn = [&context, &bu]() {
    ;
    ;
    context->executeV2(bu);
  };

  cr::benchmark::benchmark_throughput(fn, 1);

  // context->executeV2(bu);

  std::cout << "done" << std::endl;

  return 0;
}
