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

template <typename T, int N>
struct Ldg_functor {
  inline __device__ Ldg_functor(T (&fetchs)[N], const void* (&ptrs)[N]) : fetchs_(fetchs), ptrs_(ptrs) {}

  inline __device__ void load(int ii, bool p) {
    if (p) {
      ldg(fetchs_[ii], ptrs_[ii]);
    }
  }

  T (&fetchs_)[N];
  const void* (&ptrs_)[N];
};

template <int NUM_THREADS_, int BYTES_PER_ELEMENT_, int ROWS_, int COLS_, int BYTES_PER_NUM_LDGS_ = 16>
struct Gmem_tile_qkv {
  static constexpr int NUM_THREADS = NUM_THREADS_;
  static constexpr int BYTES_PER_ELEMENT = BYTES_PER_ELEMENT_;
  static constexpr int ROWS = ROWS_;
  static constexpr int COLS = COLS_;
  static constexpr int BYTES_PER_LDG = BYTES_PER_NUM_LDGS_;

  static constexpr int BYTES_PER_ROW = COLS * BYTES_PER_ELEMENT;
  // The number of threads to load a "row" of the matrix.
  static constexpr int THREADS_PER_ROW = BYTES_PER_ROW / BYTES_PER_LDG;
  // The number of "rows" loaded per LDG.
  static_assert(NUM_THREADS % THREADS_PER_ROW == 0);
  static constexpr int ROWS_PER_LDG = NUM_THREADS / THREADS_PER_ROW;
  // The number of LDGS needed to load a chunk of the Q matrix.
  static constexpr int NUM_LDGS = DivUpConstexpr(ROWS, ROWS_PER_LDG);

  // Ctor.
  template <typename BInfo>
  inline __device__ Gmem_tile_qkv(void* ptr_, const uint32_t row_stride_in_eltsconst int tidx, int actual_cols)
      : row_stride_in_bytes(row_stride_in_elts * BYTES_PER_ELEMENT),
        ptr(reinterpret_cast<char*>(ptr_)),
        tidx_(tidx),
        col_predicate((tidx % THREADS_PER_ROW) * (BYTES_PER_LDG / BYTES_PER_ELEMENT) < actual_cols) {
    base_row_ = tidx_ / THREADS_PER_ROW;
    base_col_ = tidx_ % THREADS_PER_ROW;
  }

  inline __device__ void load(const uint4 (&fetch)[NUM_LDGS]) {
    const void* ptrs[NUM_LDGS];
    uint32_t preds[NUM_LDGS];
#pragma unroll
    for (int ii = 0; ii < NUM_LDGS; ++ii) {
      int row = base_row_ + static_cast<uint32_t>(ii) * ROWS_PER_LDG;
      ptrs[ii] = ptr + row * row_stride_in_bytes;
      preds[ii] = col_predicate && row < ROWS;
    }

    Ldg_functor<uint4, NUM_LDGS> fct(fetch, ptrs);
#pragma unroll
    for (int ii = 0; ii < NUM_LDGS; ++ii) {
      fct.load(ii, preds[ii]);
    }
  }

  inline __device__ void store(const uint4 (&data)[NUM_LDGS]) {
#pragma unroll
    for (int ii = 0; ii < NUM_LDGS; ++ii) {
      int row = base_row_ + ii * ROWS_PER_LDG;
      char* ptr_ = ptr + (uint32_t)ii * ROWS_PER_LDG * row_stride_in_bytes;
      if (col_predicate && row < ROWS) {
        stg(ptr_, data[ii]);
      }
    }
  }

  const uint32_t row_stride_in_bytes;
  char* ptr;
  uint4 fetch_[NUM_LDGS];
  const int tidx_;
  const bool col_predicate;

  const int base_row_;
  const int base_col_;
};

}  // namespace cr
