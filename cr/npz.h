#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

namespace cr::npz {

class NpyArray {
 public:
  enum class ElementTypeID : int64_t {
    UNKNOW = 0,
    FLOAT16 = 1,
    FLOAT = 2,
    DOUBLE = 3,
    INT8 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
  };
  NpyArray() = delete;
  NpyArray(const std::vector<size_t>& shape, ElementTypeID type_id, std::unique_ptr<char[]> buffer)
      : type_id_(type_id), shape_(shape), buffer_(std::move(buffer)) {}
  NpyArray(const NpyArray&) = delete;
  NpyArray(NpyArray&& other) {
    type_id_ = other.type_id_;
    shape_ = other.shape_;
    buffer_ = std::move(other.buffer_);
  }

  const std::vector<size_t> getShape() const { return shape_; }
  const void* getData() const { return buffer_.get(); }
  ElementTypeID getTypeID() const { return type_id_; }

  size_t numel() const { return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<size_t>()); }
  size_t getElementByteWidth() const {
    switch (type_id_) {
      case ElementTypeID::DOUBLE:
        return 8;
      case ElementTypeID::FLOAT:
        return 4;
      case ElementTypeID::FLOAT16:
        return 2;
      case ElementTypeID::INT64:
        return 8;
      case ElementTypeID::INT32:
        return 4;
      case ElementTypeID::INT16:
        return 2;
      case ElementTypeID::INT8:
        return 1;
    }
    return 0;
  }
  static size_t getElementByteWidth(ElementTypeID type_id) {
    switch (type_id) {
      case ElementTypeID::DOUBLE:
        return 8;
      case ElementTypeID::FLOAT:
        return 4;
      case ElementTypeID::FLOAT16:
        return 2;
      case ElementTypeID::INT64:
        return 8;
      case ElementTypeID::INT32:
        return 4;
      case ElementTypeID::INT16:
        return 2;
      case ElementTypeID::INT8:
        return 1;
    }
    return 0;
  }
  size_t nbytes() const { return this->numel() * this->getElementByteWidth(); }

 private:
  ElementTypeID type_id_;
  std::vector<size_t> shape_;
  std::unique_ptr<char[]> buffer_;
};

using npz_t = std::map<std::string, NpyArray>;

npz_t load_npz(const std::string& npz_file);
void save_npz(const npz_t& npz, const std::string& npz_file);

}  // namespace cr::npz
