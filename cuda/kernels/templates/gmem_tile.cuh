#pragma once

namespace cr {

constexpr int DivUpConstexpr(int M, int N) { return (M + N - 1) / N; }

template <typename T>
inline __device__ void ldg(T& dst, const void* ptr) {
  dst = *reinterpret_cast<const T*>(ptr);
}

template <typename T>
inline __device__ void stg(void* ptr, T src) {
  *reinterpret_cast<T*>(ptr) = src;
}

template <typename ElementT_, typename LoadOrStoreT_, int NUM_THREADS_, int ROWS_, int COLS_>
struct GmemToRegsTileLoader {
  using ElementT = ElementT_;
  using LoadOrStoreT = LoadOrStoreT_;
  static constexpr int NUM_THREADS = NUM_THREADS_;
  static constexpr int ROWS = ROWS_;
  static constexpr int COLS = COLS_;

  static constexpr int BYTES_PER_ELEMENT = sizeof(ElementT);
  static constexpr int BYTES_PER_LDG = sizeof(LoadOrStoreT);

  static constexpr int BYTES_PER_ROW = COLS * BYTES_PER_ELEMENT;
  // The number of threads to load a "row" of the matrix.
  static_assert(BYTES_PER_ROW % BYTES_PER_LDG == 0);
  static constexpr int THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG;
  // The number of "rows" loaded per LDG.
  static_assert(NUM_THREADS % THREADS_PER_ROW == 0);
  static constexpr int ROWS_PER_LDG = NUM_THREADS / THREADS_PER_ROW;
  // The number of LDGS needed to load a tile of the matrix.
  static constexpr int NUM_LDGS = DivUpConstexpr(ROWS, ROWS_PER_LDG);

  // Ctor.
  inline __device__ GmemToRegsTileLoader(void* ptr_, int row_stride_in_elts, const int tidx, int actual_cols)
      : row_stride_in_bytes(row_stride_in_elts * BYTES_PER_ELEMENT),
        ptr(reinterpret_cast<char*>(ptr_)),
        tidx_(tidx),
        base_row_(tidx / THREADS_PER_ROW),
        base_col_(tidx % THREADS_PER_ROW),
        col_predicate((tidx % THREADS_PER_ROW) * BYTES_PER_LDG / BYTES_PER_ELEMENT < actual_cols) {}

  inline __device__ void load(LoadOrStoreT (&fetch)[NUM_LDGS]) {
#pragma unroll
    for (int ii = 0; ii < NUM_LDGS; ++ii) {
      int row = base_row_ + ii * ROWS_PER_LDG;
      auto current_ptr = ptr + row * row_stride_in_bytes;
      if (col_predicate && row < ROWS) {
        ldg(fetch[ii], current_ptr);
      }
    }
  }

  inline __device__ void store(const LoadOrStoreT (&data)[NUM_LDGS]) {
#pragma unroll
    for (int ii = 0; ii < NUM_LDGS; ++ii) {
      int row = base_row_ + ii * ROWS_PER_LDG;
      char* ptr_ = ptr + row * row_stride_in_bytes;
      if (col_predicate && row < ROWS) {
        stg(ptr_, data[ii]);
      }
    }
  }

  int row_stride_in_bytes;
  char* ptr;
  int tidx_;

  int base_row_;
  int base_col_;

  bool col_predicate;
};

}  // namespace cr
