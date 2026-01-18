#include <nvshmem.h>
#include <nvshmemx.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  nvshmem_init();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  int* buf = (int*)nvshmem_malloc(sizeof(int));
  *buf = 0;

  nvshmem_barrier_all();

  if (mype == 0) {
    *buf = 42;
    nvshmem_int_put(buf, buf, 1, 1);
    printf("PE %d sent value %d to PE 1\n", mype, *buf);
  }

  nvshmem_barrier_all();

  if (mype == 1) {
    printf("PE %d received value %d from PE 0\n", mype, *buf);
  }

  nvshmem_free(buf);
  nvshmem_finalize();
  return 0;
}