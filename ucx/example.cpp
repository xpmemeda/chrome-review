#include <iostream>
#include <memory>
#include <string>

#include "comm/comm.h"
#include "ucxbackend.h"

int main() {
  auto pid = fork();
  if (pid < 0) {
    printf("fork err");
    return 1;
  }

  int local_port = pid == 0 ? 6666 : 6000;
  int remote_port = 6666 + 6000 - local_port;

  // 1GB.
  std::vector<uint8_t> buffer(1ull << 30, 33);

  UcxBackend backend(comm::StringPrintf("127.0.0.1:%d", local_port));
  backend.Init();

  backend.RegisterBuffer(buffer.data(), buffer.size());

  for (size_t i = 0; i < 10; ++i) {
    comm::IntervalTiming timer;

    if (pid != 0) {
      backend.SendBytes(buffer.data(), buffer.size(), 0x888, comm::StringPrintf("127.0.0.1:%d", remote_port));
    } else {
      backend.RecvBytes(buffer.data(), buffer.size(), 0x888, comm::StringPrintf("127.0.0.1:%d", remote_port));
    }

    size_t ngbs = buffer.size() * 1e-9;
    int ms = timer.PassTime();
    double throughput = ngbs / (ms * 1e-3);

    LogHead("send-recv {} GB cost {} ms throughput {:.2f} GB/s", ngbs, ms, throughput);
  }

  return 0;
}
