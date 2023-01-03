#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

class SpinLock {
  std::atomic<bool> spin;

public:
  SpinLock() : spin(false) {}
  SpinLock(const SpinLock &) = delete;
  SpinLock(SpinLock &&) = delete;

  SpinLock &operator=(const SpinLock &) = delete;
  SpinLock &operator=(SpinLock &&) = delete;

  void lock() {
    bool expire;
    do {
      expire = false;
    } while (!spin.compare_exchange_weak(expire, true));
  }

  bool try_lock() {
    bool expire = false;
    return spin.compare_exchange_weak(expire, true);
  }

  void unlock() { spin.store(false); }
};

void fun(int i) {
  static SpinLock mutex;
  static int sum = 0;
  std::lock_guard<SpinLock> lck(mutex);
  sum += i;
  std::cout << i << ", sum = " << sum << std::endl;
}

int main() {
  std::vector<std::thread> ths;
  for (int i = 0; i < 10; ++i) {
    ths.emplace_back(fun, i);
  }
  for (int i = 0; i < 10; ++i) {
    ths[i].join();
  }
  return 0;
}