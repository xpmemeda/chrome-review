#include <transfer_engine.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

using namespace mooncake;
using namespace std::chrono_literals;

// 小工具：轮询一个 batch 的第 task_id 个任务，直到完成或超时
bool wait_for_completion(
    TransferEngine& engine, BatchID batch_id, size_t task_id, int timeout_ms = 5000, int poll_interval_ms = 10) {
  using clock = std::chrono::steady_clock;
  auto begin = clock::now();

  while (true) {
    TransferStatus st;
    mooncake::Status rc = engine.getTransferStatus(batch_id, task_id, st);
    if (rc != mooncake::Status::OK()) {
      std::cerr << "getTransferStatus failed, rc=" << rc << "\n";
      engine.freeBatchID(batch_id);
      return false;
    }

    if (st.s == mooncake::TransferStatusEnum::COMPLETED) {
      engine.freeBatchID(batch_id);
      return true;
    }
    if (st.s == mooncake::TransferStatusEnum::FAILED || st.s == mooncake::TransferStatusEnum::INVALID) {
      std::cerr << "transfer failed, status=" << static_cast<int>(st.s)
                << ", transferred_bytes=" << st.transferred_bytes << "\n";
      engine.freeBatchID(batch_id);
      return false;
    }

    auto now = clock::now();
    if (timeout_ms > 0 && std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count() >= timeout_ms) {
      std::cerr << "wait_for_completion timeout\n";
      engine.freeBatchID(batch_id);
      return false;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(poll_interval_ms));
  }
}

int main() {
  // 1. 准备 metadata server 地址（这里假设你本机开了 etcd）
  const std::string metadata_server = "http://127.0.0.1:8080/metadata";

  // 2. 创建两个 TransferEngine 实例，相当于同进程内的“nodeA”和“nodeB”:contentReference[oaicite:2]{index=2}
  TransferEngine engineA(true);
  TransferEngine engineB(true);

  int rc = 0;
  mooncake::Status st = mooncake::Status::OK();

  // local_server_name 就是 RAM Segment 的名字
  rc = engineA.init(metadata_server, "nodeA");
  if (rc != 0) {
    std::cerr << "engineA.init failed, rc=" << rc << "\n";
    return 1;
  }
  rc = engineB.init(metadata_server, "nodeB");
  if (rc != 0) {
    std::cerr << "engineB.init failed, rc=" << rc << "\n";
    return 1;
  }

  // 3. 在同一进程里分配几块 DRAM buffer
  const size_t kSize = 1 << 20;  // 1 MiB
  void* bufA = std::malloc(kSize);
  void* bufB = std::malloc(kSize);   // engineB 的目标 buffer
  void* bufB2 = std::malloc(kSize);  // engineB 本地用来再从 nodeA 读一次

  if (!bufA || !bufB || !bufB2) {
    std::cerr << "malloc failed\n";
    return 1;
  }

  // 初始化一下内容
  std::memset(bufA, 0xAB, kSize);
  std::memset(bufB, 0x00, kSize);
  std::memset(bufB2, 0x00, kSize);

  // 4. 把这些 buffer 注册进各自的 TransferEngine RAM Segment 里:contentReference[oaicite:3]{index=3}
  // location 用 "*" 让 TE 自动识别是 CPU 内存
  rc = engineA.registerLocalMemory(bufA, kSize, "*", /*remote_accessible=*/true, true);
  if (rc != 0) {
    std::cerr << "engineA.registerLocalMemory failed, rc=" << rc << "\n";
    return 1;
  }
  rc = engineB.registerLocalMemory(bufB, kSize, "*", /*remote_accessible=*/true, true);
  if (rc != 0) {
    std::cerr << "engineB.registerLocalMemory failed, rc=" << rc << "\n";
    return 1;
  }
  rc = engineB.registerLocalMemory(bufB2, kSize, "*", /*remote_accessible=*/false, true);
  if (rc != 0) {
    std::cerr << "engineB.registerLocalMemory(bufB2) failed, rc=" << rc << "\n";
    return 1;
  }

  // 5. 在同一进程里，engineA 打开 "nodeB" 这个 Segment，engineB 打开 "nodeA"
  auto segB = engineA.openSegment("nodeB");
  if ((int64_t)segB < 0) {
    std::cerr << "engineA.openSegment(nodeB) failed, segB " << (int64_t)segB << "\n";
    return 1;
  }

  auto segA = engineB.openSegment("nodeA");
  if ((int64_t)segA < 0) {
    std::cerr << "engineB.openSegment(nodeA) failed, segA " << (int64_t)segA << "\n";
    return 1;
  }

  // 6. engineA 做一次 WRITE：bufA -> nodeB.bufB:contentReference[oaicite:4]{index=4}
  {
    BatchID batch_id = engineA.allocateBatchID(1);
    if (batch_id < 0) {
      std::cerr << "engineA.allocateBatchID failed\n";
      return 1;
    }

    mooncake::TransferRequest req;
    req.opcode = mooncake::TransferRequest::WRITE;
    req.source = bufA;  // 本地起点
    req.target_id = static_cast<mooncake::SegmentID>(segB);
    // 同进程共享虚拟地址空间，所以这里直接用 bufB 的地址做 target_offset
    req.target_offset = reinterpret_cast<size_t>(bufB);
    req.length = kSize;

    std::vector<mooncake::TransferRequest> entries{req};
    st = engineA.submitTransfer(batch_id, entries);
    if (st != mooncake::Status::OK()) {
      std::cerr << "engineA.submitTransfer failed, st=" << st << "\n";
      return 1;
    }

    bool ok = wait_for_completion(engineA, batch_id, /*task_id=*/0);
    std::cout << "[engineA] WRITE bufA -> nodeB.bufB: " << (ok ? "OK" : "FAIL") << "\n";
  }

  // 简单检查一下 bufB 里的内容是否变成 0xAB
  std::cout << "bufB first byte after WRITE = 0x" << std::hex
            << static_cast<int>(reinterpret_cast<unsigned char*>(bufB)[0]) << std::dec << "\n";

  // 7. engineB 再做一次 READ：从 nodeA.bufA 读到 bufB2
  {
    BatchID batch_id = engineB.allocateBatchID(1);
    if (batch_id < 0) {
      std::cerr << "engineB.allocateBatchID failed\n";
      return 1;
    }

    mooncake::TransferRequest req;
    req.opcode = mooncake::TransferRequest::READ;
    req.source = bufB2;  // 本地接收 buffer
    req.target_id = static_cast<mooncake::SegmentID>(segA);
    req.target_offset = reinterpret_cast<size_t>(bufA);  // nodeA 那边的地址
    req.length = kSize;

    std::vector<mooncake::TransferRequest> entries{req};
    st = engineB.submitTransfer(batch_id, entries);
    if (st != mooncake::Status::OK()) {
      std::cerr << "engineB.submitTransfer failed, rc=" << rc << "\n";
      return 1;
    }

    bool ok = wait_for_completion(engineB, batch_id, /*task_id=*/0);
    std::cout << "[engineB] READ nodeA.bufA -> bufB2: " << (ok ? "OK" : "FAIL") << "\n";
  }

  // 检查 bufB2 的前几个字节
  std::cout << "bufB2 first 4 bytes after READ: ";
  for (int i = 0; i < 4; ++i) {
    std::cout << "0x" << std::hex << static_cast<int>(reinterpret_cast<unsigned char*>(bufB2)[i]) << " ";
  }
  std::cout << std::dec << "\n";

  std::free(bufA);
  std::free(bufB);
  std::free(bufB2);

  std::cout << "done.\n";
  return 0;
}
