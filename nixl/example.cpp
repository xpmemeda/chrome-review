#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <nixl.h>
#include <sys/time.h>
#include <utils/serdes/serdes.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nixlbackend.h"

int main() {
  auto pid = fork();
  if (pid < 0) {
    printf("fork err");
    return 1;
  }

  auto x = at::empty({10, 1024, 1024}, at::ScalarType::Float);
  if (pid == 0) {
    x.fill_(0);
  } else {
    x.fill_(1);
  }

  int local_port = pid == 0 ? 6666 : 6000;
  int remote_port = 6666 + 6000 - local_port;

  NixlBackend backend(comm::StringPrintf("127.0.0.1:%d", local_port));

  backend.registerBuffer(x.data_ptr(), x.nbytes());

  for (size_t i = 0; i < 10; ++i) {
    comm::IntervalTiming timer;

    if (pid != 0) {
      backend.send(x.data_ptr(), x.nbytes(), "0", comm::StringPrintf("127.0.0.1:%d", remote_port));
    } else {
      backend.recv(x.data_ptr(), x.nbytes(), "0", comm::StringPrintf("127.0.0.1:%d", remote_port));
    }

    double ngbs = x.nbytes() * 1e-9;
    int ms = timer.PassTime();
    double throughput = ngbs / (ms * 1e-3);

    LogHead("send-recv {:.2f} GB cost {} ms throughput {:.2f} GB/s", ngbs, ms, throughput);
  }

  if (pid == 0) {
    for (size_t i = 0; i < 5; ++i) {
      at::allclose(x[i * 2], at::ones({1024, 1024}, at::ScalarType::Float));
      at::allclose(x[i * 2 + 1], at::zeros({1024, 1024}, at::ScalarType::Float));
    }
    LogHead("test pass.");
  }

  return 0;
}