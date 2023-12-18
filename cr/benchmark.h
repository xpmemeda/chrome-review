#pragma once

#include <functional>

namespace cr::benchmark {

void benchmark_throughput(std::function<void()> func, int thread_count);

}  // namespace cr::benchmark
