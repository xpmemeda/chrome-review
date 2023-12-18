#include "benchmark.h"

#include <stdio.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

namespace cr::benchmark {

class Duration {
  std::chrono::steady_clock::duration du;

 public:
  Duration(std::chrono::steady_clock::duration d) : du(d) {}

  unsigned long microseconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::microseconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long milliseconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::milliseconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long seconds() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::seconds>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long minutes() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::minutes>(du).count();
    return std::max(0lu, cost);
  }

  unsigned long hours() const {
    unsigned long cost = std::chrono::duration_cast<std::chrono::hours>(du).count();
    return std::max(0lu, cost);
  }
};

class Coster {
  std::chrono::system_clock::time_point start;

 public:
  Coster() : start(std::chrono::system_clock::now()) {}

  void reset() { start = std::chrono::system_clock::now(); }

  Duration lap() const {
    auto now = std::chrono::system_clock::now();
    return Duration(now - start);
  }
};

void benchmark_throughput(std::function<void()> func, int thread_count) {
  std::atomic<size_t> process_cnt(0);
  std::atomic<size_t> total_latency(0);  // micro seconds
  std::atomic<size_t> max_latency(0);    // micro seconds

  std::vector<std::thread> workers;
  for (int i = 0; i < thread_count; ++i) {
    workers.emplace_back([&]() {
      while (true) {
        Coster coster;
        func();
        size_t latency = coster.lap().microseconds();
        process_cnt += 1;
        total_latency += latency;
        if (latency > max_latency.load()) {
          max_latency.store(latency);
        }
      }
    });
  }

  while (true) {
    size_t old_process_cnt = process_cnt.load();
    size_t old_total_latency = total_latency.load();
    max_latency.store(0);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    size_t new_process_cnt = process_cnt.load();
    size_t new_total_latency = total_latency.load();

    double throughput = static_cast<double>(new_process_cnt - old_process_cnt) / 10;
    double avgcost =
        static_cast<double>(new_total_latency - old_total_latency) / (new_process_cnt - old_process_cnt) / 1000;
    double maxcost = static_cast<double>(max_latency.load()) / 1000;
    printf("qps: %f, avgcost: %f, maxcost: %f\n", throughput, avgcost, maxcost);
  }
}

}  // namespace cr::benchmark
