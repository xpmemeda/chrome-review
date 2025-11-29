// mooncake_backend_demo.cpp
//
// g++ -std=c++17 mooncake_backend_demo.cpp -ltransfer_engine -lpthread -o mooncake_demo
//
// 需要：
//   - Mooncake 编译好，包含 transfer_engine.h
//   - 链接 libtransfer_engine.a
//
// 运行示例：
//   1. 启动元数据服务（例如 etcd）：
//      etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://10.0.0.1:2379
//
//   2. 在节点 A 上启动 receiver：
//      ./mooncake_demo --mode=recv --metadata=etcd://10.0.0.1:2379 --name=nodeA --peer=nodeB
//
//   3. 在节点 B 上启动 sender：
//      ./mooncake_demo --mode=send --metadata=etcd://10.0.0.1:2379 --name=nodeB --peer=nodeA
//
// 注意：demo 里假设双方 buffer 虚拟地址一致，只是为了简单说明 TE 的用法。

#include <transfer_engine.h>  // 来自 mooncake-transfer-engine/include
#include <chrono>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using namespace mooncake;

class MooncakeBackend {
 public:
  struct Endpoint {
    std::string segment_name;                  // 对端的 server_name / segment_name
    TransferEngine::SegmentHandle segment_id;  // openSegment 返回
  };

  // metadata_server: "etcd://10.0.0.1:2379" / "redis://..." / "http://host:8080/metadata"
  // local_server_name: 本机在集群中的唯一名字（TE 的 RAM Segment 名）:contentReference[oaicite:2]{index=2}
  // location: registerLocalMemory 的 location，比如 "cpu:0" / "cuda:0" /
  // "*"（自动识别）:contentReference[oaicite:3]{index=3}
  MooncakeBackend(
      const std::string& metadata_server, const std::string& local_server_name, const std::string& location = "*")
      : metadata_server_(metadata_server), local_server_name_(local_server_name), location_(location) {
    int rc = engine_.init(metadata_server_, local_server_name_);
    if (rc != 0) {
      throw std::runtime_error("TransferEngine::init failed, rc=" + std::to_string(rc));
    }
  }

  ~MooncakeBackend() {
    // TransferEngine 析构时会自动清理自身的 meta 信息:contentReference[oaicite:4]{index=4}
    // 这里无需额外操作
  }

  // 注册一块本地内存，让 TE 可以通过 RDMA/TCP 访问它:contentReference[oaicite:5]{index=5}
  void registerBuffer(void* buffer, size_t size, bool remote_accessible = true) {
    int rc = engine_.registerLocalMemory(buffer, size, location_, remote_accessible);
    if (rc != 0) {
      throw std::runtime_error("registerLocalMemory failed, rc=" + std::to_string(rc));
    }
  }

  // 从本地 buffer 向 peer 的对应地址写数据（点对点发送）
  // 这里我们简单地假设对端也在相同虚拟地址上分配并 register 了同样大小的 buffer。
  bool send(void* buffer, size_t size, const std::string& tag, const std::string& peer_address) {
    (void)tag;  // demo 里没用到 tag，你可以自己接上 DistStore/etcd 做 tag->addr 映射

    Endpoint* ep = getPeerEndpoint(peer_address);
    if (!ep) {
      std::cerr << "send: getPeerEndpoint failed\n";
      return false;
    }

    TransferEngine::BatchID batch_id = engine_.allocateBatchID(1);
    if (batch_id < 0) {
      std::cerr << "send: allocateBatchID failed\n";
      return false;
    }

    TransferEngine::TransferRequest req;
    req.opcode = TransferEngine::TransferRequest::WRITE;  // local -> remote:contentReference[oaicite:6]{index=6}
    req.source = buffer;
    req.target_id = ep->segment_id;
    req.target_offset = reinterpret_cast<size_t>(buffer);  // 简化假设：对端同地址
    req.length = size;

    std::vector<TransferEngine::TransferRequest> entries{req};
    int rc = engine_.submitTransfer(batch_id, entries);
    if (rc != 0) {
      std::cerr << "send: submitTransfer failed, rc=" << rc << "\n";
      engine_.freeBatchID(batch_id);
      return false;
    }

    return waitForCompletion(batch_id, /*task_id=*/0);
  }

  // 从 peer 的 buffer 读数据到本地 buffer（点对点接收）
  bool recv(void* buffer, size_t size, const std::string& tag, const std::string& peer_address) {
    (void)tag;

    Endpoint* ep = getPeerEndpoint(peer_address);
    if (!ep) {
      std::cerr << "recv: getPeerEndpoint failed\n";
      return false;
    }

    TransferEngine::BatchID batch_id = engine_.allocateBatchID(1);
    if (batch_id < 0) {
      std::cerr << "recv: allocateBatchID failed\n";
      return false;
    }

    TransferEngine::TransferRequest req;
    req.opcode = TransferEngine::TransferRequest::READ;  // remote -> local:contentReference[oaicite:7]{index=7}
    req.source = buffer;
    req.target_id = ep->segment_id;
    req.target_offset = reinterpret_cast<size_t>(buffer);  // 简化假设：对端同地址
    req.length = size;

    std::vector<TransferEngine::TransferRequest> entries{req};
    int rc = engine_.submitTransfer(batch_id, entries);
    if (rc != 0) {
      std::cerr << "recv: submitTransfer failed, rc=" << rc << "\n";
      engine_.freeBatchID(batch_id);
      return false;
    }

    return waitForCompletion(batch_id, /*task_id=*/0);
  }

 private:
  Endpoint* getPeerEndpoint(const std::string& peer_address) {
    std::lock_guard<std::mutex> guard(mu_);
    auto it = endpoints_.find(peer_address);
    if (it != endpoints_.end()) {
      return &it->second;
    }

    // peer_address 就是对端 init 时的 local_server_name，也就是 segment_name:contentReference[oaicite:8]{index=8}
    TransferEngine::SegmentHandle seg = engine_.openSegment(peer_address);
    if (seg < 0) {
      std::cerr << "openSegment failed for " << peer_address << "\n";
      return nullptr;
    }

    Endpoint ep{peer_address, seg};
    auto [iter, ok] = endpoints_.emplace(peer_address, ep);
    return ok ? &iter->second : nullptr;
  }

  bool waitForCompletion(
      TransferEngine::BatchID batch_id, size_t task_id, int timeout_ms = 10000, int poll_interval_ms = 10) {
    using namespace std::chrono;
    auto start = steady_clock::now();

    while (true) {
      TransferEngine::TransferStatus st;
      int rc = engine_.getTransferStatus(batch_id, task_id, st);
      if (rc != 0) {
        std::cerr << "getTransferStatus failed, rc=" << rc << "\n";
        engine_.freeBatchID(batch_id);
        return false;
      }

      if (st.s == TransferEngine::TaskStatus::COMPLETED) {
        engine_.freeBatchID(batch_id);
        return true;
      }

      if (st.s == TransferEngine::TaskStatus::FAILED || st.s == TransferEngine::TaskStatus::INVALID) {
        std::cerr << "transfer failed, status=" << static_cast<int>(st.s) << ", transferred=" << st.transferred << "\n";
        engine_.freeBatchID(batch_id);
        return false;
      }

      auto now = steady_clock::now();
      if (timeout_ms > 0 && duration_cast<milliseconds>(now - start).count() >= timeout_ms) {
        std::cerr << "waitForCompletion timed out\n";
        engine_.freeBatchID(batch_id);
        return false;
      }

      std::this_thread::sleep_for(milliseconds(poll_interval_ms));
    }
  }

 private:
  std::string metadata_server_;
  std::string local_server_name_;
  std::string location_;

  TransferEngine engine_;
  std::unordered_map<std::string, Endpoint> endpoints_;
  std::mutex mu_;
};

// 一个很简单的 main，用来在两台机器之间传 1MB 数据做 demo
int main(int argc, char** argv) {
  std::string mode = "send";  // "send" or "recv"
  std::string metadata_server = "etcd://127.0.0.1:2379";
  std::string local_name = "nodeA";
  std::string peer_name = "nodeB";

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto get_val = [&](const std::string& prefix) -> std::string {
      if (arg.rfind(prefix, 0) == 0) {
        return arg.substr(prefix.size());
      }
      return {};
    };
    if (auto v = get_val("--mode="); !v.empty()) mode = v;
    if (auto v = get_val("--metadata="); !v.empty()) metadata_server = v;
    if (auto v = get_val("--name="); !v.empty()) local_name = v;
    if (auto v = get_val("--peer="); !v.empty()) peer_name = v;
  }

  try {
    MooncakeBackend backend(metadata_server, local_name, "*");

    const size_t kSize = 1 << 20;  // 1 MiB
    void* buf = std::malloc(kSize);
    if (!buf) {
      std::cerr << "malloc failed\n";
      return 1;
    }

    backend.registerBuffer(buf, kSize, /*remote_accessible=*/true);

    if (mode == "send") {
      // 填充数据然后发过去
      std::memset(buf, 0x5A, kSize);  // 0x5A = 'Z'
      if (!backend.send(buf, kSize, "demo_tag", peer_name)) {
        std::cerr << "send failed\n";
        return 1;
      }
      std::cout << "send done\n";
    } else if (mode == "recv") {
      // 从对端读到本地 buffer
      std::memset(buf, 0, kSize);
      if (!backend.recv(buf, kSize, "demo_tag", peer_name)) {
        std::cerr << "recv failed\n";
        return 1;
      }
      std::cout << "recv done, first byte=" << std::hex << static_cast<int>(reinterpret_cast<unsigned char*>(buf)[0])
                << std::dec << "\n";
    } else {
      std::cerr << "unknown mode: " << mode << "\n";
      return 1;
    }

    std::free(buf);
  } catch (const std::exception& ex) {
    std::cerr << "exception: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
