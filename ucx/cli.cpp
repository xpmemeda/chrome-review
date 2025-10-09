#include <iostream>

#include "comm/comm.h"
#include "ucxtx.h"

int main() {
  UcxTransport tx;

  std::string server = "127.0.0.1:6000";
  std::vector<std::uint8_t> buf(1024 * 1024 * 1024, 8);

  for (int i = 0; i < 10; ++i) {
    comm::IntervalTiming timer;

    if (!tx.send(server, buf)) {
      std::cerr << "send failed\n";
      return 1;
    }

    std::vector<std::uint8_t> reply;
    if (!tx.recv(server, reply)) {
      std::cerr << "recv failed\n";
      return 1;
    }

    std::cout << "Client got reply size=" << reply.size() << "\n";

    printf("8GB cost %d ms\n", timer.PassTime());
  }

  sleep(1000);
  return 0;
}
