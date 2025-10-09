#include "ucxtx.h"

int main() {
  UcxTransport tx;

  if (!tx.listen("0.0.0.0:6000")) {
    std::cerr << "listen failed\n";
    return 1;
  }

  std::cout << "Server listening on 0.0.0.0:6000\n";

  while (true) {
    auto peers = tx.list_peers();
    for (auto& peer : peers) {
      std::vector<std::uint8_t> recv_buf;
      if (tx.recv(peer, recv_buf)) {
        std::cout << "Got msg from " << peer << ", size=" << recv_buf.size() << "\n";
        tx.send(peer, recv_buf);
      }
    }

    usleep(1000);
  }
}