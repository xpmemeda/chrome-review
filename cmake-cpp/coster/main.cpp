#include <chrono>
#include <iostream>
#include <vector>

template <typename T> int measuring_cost(T &&func) {
  auto t1 = std::chrono::system_clock::now();
  for (int i = 0; i < 1000; ++i)
    func(i);
  auto t2 = std::chrono::system_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
}

int main() {
  {
    // 60
    int cost =
        measuring_cost([](int) { auto t = std::chrono::system_clock::now(); });
    std::cout << cost << std::endl;
  }

  {
    // 6
    float x = 18.09;
    int cost = measuring_cost([&](int) { x *= 1.1793; });
    std::cout << cost << std::endl;
  }

  {
    // malloc: 2410
    // free: 1470
    std::vector<void *> pointers(1000, nullptr);
    int costMalloc =
        measuring_cost([&pointers](int i) { pointers[i] = malloc(1 << 20); });
    int costFree = measuring_cost([&pointers](int i) { free(pointers[i]); });
    std::cout << "malloc: " << costMalloc << "\nfree:" << costFree << std::endl;
  }

  {
    // push_back: 26
    // insert: 233
    std::vector<int> values;
    int backInsertionCost =
        measuring_cost([&values](int i) { values.push_back(i); });
    std::vector<int>().swap(values);
    int frontInsertionCost =
        measuring_cost([&values](int i) { values.insert(values.begin(), i); });
    std::vector<int>().swap(values);
    std::cout << "push_back: " << backInsertionCost
              << "\ninsert: " << frontInsertionCost << std::endl;
  }

  return 0;
}
