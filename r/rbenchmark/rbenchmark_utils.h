#pragma once

#include <functional>

namespace rbenchmark {

void benchmark_throughput(std::function<void()> func, int thread_count);

}