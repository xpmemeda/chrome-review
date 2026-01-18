#include <ATen/Dispatch.h>
#include <ATen/Tensor.h>
#include <cuda_runtime.h>

#include "dbgrec.cuh"
#include "module.h"

namespace {

struct DbgRec {
  int32_t bid;
  int32_t tid;

  void* data;
  size_t index;
};

template <typename T, typename Rec>
__global__ void oob_fill_kernel(T* data, size_t size, T value, DbgState<Rec>* dbg_state) {
  size_t oob_offset = 0xffffffffffff;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;

  for (size_t i = idx + oob_offset; i < size + oob_offset; i += stride) {
    if (i >= size) {
      DbgRec rec;
      rec.bid = blockIdx.x;
      rec.tid = threadIdx.x;
      rec.data = data;
      rec.index = i;
      dbg_record(dbg_state, rec);

      // asm("trap;");
      return;
    }

    data[i] = value;
  }
}

void oob_fill(at::Tensor& data, at::Tensor& value) {
  size_t size = data.numel();
  size_t threads = 256;
  size_t blocks = (size + threads - 1) / threads;

  AT_DISPATCH_ALL_TYPES(data.scalar_type(), "oob_fill_kernel", [&] {
    DbgState<DbgRec>* dbg_state = nullptr;
    init_dbg_state_on_device(&dbg_state);

    oob_fill_kernel<scalar_t>
        <<<blocks, threads>>>(data.data_ptr<scalar_t>(), size, *value.data_ptr<scalar_t>(), dbg_state);

    fetch_and_print_dbg_state_on_host(dbg_state, [](const DbgRec& r) {
      printf("[dbg] record: bid=%d, tid=%d, data=%p, index=%llu\n", r.bid, r.tid, r.data, (unsigned long long)r.index);
    });
  });

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel launch failed: ") + cudaGetErrorString(err));
  }
}

static cr::Register _([](pybind11::module& m) { m.def("oob_fill", &oob_fill); });

}  // namespace
