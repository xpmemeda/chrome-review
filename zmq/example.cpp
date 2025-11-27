#include <sys/time.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "comm/comm.h"
#include "zmqbackend.h"

int main() {
  auto pid = fork();
  if (pid < 0) {
    printf("fork err");
    return 1;
  }
  // int pid = 0;

  int local_port = pid == 0 ? 6666 : 6000;
  int remote_port = 6666 + 6000 - local_port;

  // 1GB.
  std::vector<uint8_t> buffer(1ull << 30, 33);

  ZmqBackend backend(comm::StringPrintf("127.0.0.1:%d", local_port));

  backend.registerBuffer(buffer.data(), buffer.size());

  for (size_t i = 0; i < 10; ++i) {
    comm::IntervalTiming timer;

    if (pid != 0) {
      backend.send(buffer.data(), buffer.size(), "0", comm::StringPrintf("127.0.0.1:%d", remote_port));
    } else {
      backend.recv(buffer.data(), buffer.size(), "0", comm::StringPrintf("127.0.0.1:%d", remote_port));
    }

    size_t ngbs = buffer.size() * 1e-9;
    int ms = timer.PassTime();
    double throughput = ngbs / (ms * 1e-3);

    LogHead("send-recv {} GB cost {} ms throughput {:.2f} GB/s", ngbs, ms, throughput);
  }

  return 0;
}