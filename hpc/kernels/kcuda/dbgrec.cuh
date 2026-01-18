#pragma once

#include <cuda_runtime.h>
#include <cstdint>

template <typename Rec>
struct DbgState {
  int32_t flag;  // 0: inactive, 1: active
  Rec rec;       // user-defined record
};

template <typename Rec>
__device__ __forceinline__ void dbg_record(DbgState<Rec>* state, const Rec& rec) {
  if (atomicCAS(&state->flag, 0, 1) == 0) {
    state->rec = rec;
    __threadfence_system();
  }
}

template <typename Rec>
void init_dbg_state_on_device(DbgState<Rec>** dbg_state_device) {
  cudaMalloc(dbg_state_device, sizeof(DbgState<Rec>));
  cudaMemset(*dbg_state_device, 0, sizeof(DbgState<Rec>));
}

template <typename Rec, typename F>
void fetch_and_print_dbg_state_on_host(const DbgState<Rec>* dbg_state_device, F f) {
  DbgState<Rec> dbg_state_host;
  cudaMemcpy(&dbg_state_host, dbg_state_device, sizeof(DbgState<Rec>), cudaMemcpyDeviceToHost);

  if (dbg_state_host.flag == 0) {
    printf("[dbg] No dbg record.\n");
    return;
  }

  f(dbg_state_host.rec);
}
