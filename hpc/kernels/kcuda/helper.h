#include <ATen/Tensor.h>
#include <unordered_map>

namespace comm {

class CudaMappedTensorManager {
 public:
  ~CudaMappedTensorManager();

  void _register(at::Tensor& x);
  void _release(at::Tensor& x);

  void* get(void* h_ptr);
  static CudaMappedTensorManager* get();

 private:
  std::unordered_map<void*, void*> h2ds_;
};

}  // namespace comm