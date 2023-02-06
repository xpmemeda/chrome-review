#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include <vector>

std::atomic<bool> keep_running{true};

void sig_handler(int sig) {
  if (sig == SIGINT)
    keep_running.store(false);
}

void func() {
  while (keep_running.load()) {
    std::cout << std::this_thread::get_id() << ": running..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
}

int main() {
  signal(SIGINT, sig_handler);

  std::vector<std::thread> ths;
  for (int i = 0; i < 1; ++i) {
    ths.emplace_back(func);
  }

  while (keep_running.load()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  std::cout << "Terminated 111" << std::endl;
  for (int i = 0; i < 1; ++i) {
    ths[i].join();
  }
  std::cout << "Terminated" << std::endl;
  return 0;
}