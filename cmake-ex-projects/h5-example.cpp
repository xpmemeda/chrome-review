#include <algorithm>
#include <highfive/highfive.hpp>
#include <iostream>
#include <string>
#include <vector>

inline std::string h5_dtype_to_aten(const HighFive::DataType& dt) {
  using DC = HighFive::DataTypeClass;
  const auto cls = dt.getClass();
  const auto sz = dt.getSize();

  if (cls == DC::Float) {
    if (sz == 2) return "at::kHalf";
    if (sz == 4) return "at::kFloat";
    if (sz == 8) return "at::kDouble";
  } else if (cls == DC::Integer) {
    // HighFive/HDF5 可进一步区分 signed/unsigned，这里只用 size 粗分；
    // 若你需要严格区分 uint32/uint64，可再补：H5Tget_sign(...)
    if (sz == 1) return "at::kByte";  // int8/uint8 都先当 Byte（uint8）
    if (sz == 4) return "at::kInt";
    if (sz == 8) return "at::kLong";
  }

  throw std::runtime_error("Unsupported HDF5 dtype (class/size not mapped).");
}

int main() {
  HighFive::File file("demo.h5", HighFive::File::ReadOnly);
  auto dset = file.getDataSet("x");

  std::vector<size_t> dims = dset.getSpace().getDimensions();
  // std::cout << dset. << std::endl;
  std::cout << h5_dtype_to_aten(dset.getDataType()) << std::endl;
  ;

  std::cout << "c++ read dims: [";
  for (size_t i = 0; i < dims.size(); ++i) std::cout << dims[i] << (i + 1 == dims.size() ? "" : ", ");
  std::cout << "]\n";

  // 计算元素总数
  size_t total = 1;
  for (auto d : dims) total *= d;

  std::vector<float> data(total);
  dset.read_raw(data.data());

  std::cout << "elements=" << data.size() << "\n";

  // 打印前几个元素
  size_t nprint = std::min<size_t>(data.size(), 8);
  for (size_t i = 0; i < nprint; ++i) {
    std::cout << data[i] << (i + 1 == nprint ? "\n" : ", ");
  }

  // 如果你想按 (i,j) 访问（假设是 2D）
  if (dims.size() == 2) {
    size_t H = dims[0], W = dims[1];
    auto at = [&](size_t i, size_t j) -> float& { return data[i * W + j]; };
    std::cout << "at(2,3)=" << at(2, 3) << "\n";
  }

  return 0;
}