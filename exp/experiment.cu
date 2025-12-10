#include <stdio.h>

__global__ void smem_bad() {
  __shared__ int s[32];

  int tid = threadIdx.x;

  s[tid] = tid;

  __syncthreads();
  if (tid == 0) {
    printf("done\n");
  }
}

int main() {
  smem_bad<<<1, 64>>>();
  cudaDeviceSynchronize();
}
