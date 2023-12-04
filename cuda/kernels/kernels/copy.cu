#include "copy.h"

#include "cuda_fp16.h"
#include "cuda_runtime.h"

// #include "cutlass/coord.h"
// #include "cutlass/layout/pitch_linear.h"
// #include "cutlass/transform/pitch_linear_thread_map.h"
// #include "cutlass/transform/threadblock/predicated_tile_iterator.h"

// #include "../templates/gmem_tile.cuh"
// #include "../templates/print.cuh"

// namespace cr {

// __global__ void load_print(void* ptr) {
//   using GmemTile = GmemTileT<half, uint4, 32, 32, 32>;
//   GmemTile gmem_tile(ptr, 32, threadIdx.x, 32);
//   uint4 fetchs[GmemTile::NUM_LDGS];
//   printf("%d\n", threadIdx.x);
//   gmem_tile.load(fetchs);
//   auto x = fetchs[0];
//   print_half(x);
// }

// template <typename Iterator>
// __global__ void copy(typename Iterator::Params dst_params, typename Iterator::Element* dst_pointer,
//     typename Iterator::Params src_params, typename Iterator::Element* src_pointer, cutlass::Coord<2> extent) {
//   Iterator dst_iterator(dst_params, dst_pointer, extent, threadIdx.x);
//   Iterator src_iterator(src_params, src_pointer, extent, threadIdx.x);

//   // PredicatedTileIterator uses PitchLinear layout and therefore takes in a PitchLinearShape.
//   // The contiguous dimension can be accessed via Iterator::Shape::kContiguous and the strided
//   // dimension can be accessed via Iterator::Shape::kStrided
//   int iterations = (extent[1] + Iterator::Shape::kStrided - 1) / Iterator::Shape::kStrided;

//   typename Iterator::Fragment fragment;

//   for (size_t i = 0; i < fragment.size(); ++i) {
//     fragment[i] = 0;
//   }

//   src_iterator.load(fragment);
//   dst_iterator.store(fragment);

//   ++src_iterator;
//   ++dst_iterator;

//   for (; iterations > 1; --iterations) {
//     src_iterator.load(fragment);
//     dst_iterator.store(fragment);

//     ++src_iterator;
//     ++dst_iterator;
//   }
// }

// void copy_v1(torch::Tensor& x, torch::Tensor& y) { load_print<<<1, 32>>>(x.data_ptr()); }

// void copy_v2(torch::Tensor& x, torch::Tensor& y) {
//   using Shape = cutlass::layout::PitchLinearShape<64, 4>;
//   using Layout = cutlass::layout::PitchLinear;
//   using Element = int;
//   int const kThreads = 32;
//   using ThreadMap = cutlass::transform::PitchLinearStripminedThreadMap<Shape, kThreads>;
//   using Iterator = cutlass::transform::threadblock::PredicatedTileIterator<Shape, Element, Layout, 1, ThreadMap>;

//   cutlass::Coord<2> copy_extent = cutlass::make_Coord(x.size(0), x.size(1));

//   auto layout = Layout::packed(copy_extent);
//   typename Iterator::Params dst_params(layout);
//   typename Iterator::Params src_params(layout);

//   auto src_pointer = reinterpret_cast<Iterator::Element*>(x.data_ptr());
//   auto dst_pointer = reinterpret_cast<Iterator::Element*>(y.data_ptr());

//   dim3 block(kThreads, 1);
//   dim3 grid(1, 1);
//   copy<Iterator><<<grid, block>>>(dst_params, src_pointer, src_params, dst_pointer, copy_extent);
// }

// }  // namespace cr
