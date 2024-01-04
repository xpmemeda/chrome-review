#include "farmhash.h"
#include "fingerprintcat64.h"
#include <iostream>
#include <string>
#include <vector>

int main() {
  // equal to ``StringToHashBucketFast``
  std::cout << util::Fingerprint64("a_X_1_X_+", 9) % 100 << std::endl;
  std::cout << util::Fingerprint64("a_X_1_X_-", 9) % 100 << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  std::cout << FingerprintCat64(956888297470,
                                util::Fingerprint64("a_X_1_X_+", 9)) %
                   100
            << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  auto a = util::Fingerprint64("a", 1);
  auto one = util::Fingerprint64("1", 1);
  auto plus = util::Fingerprint64("+", 1);
  std::vector<uint64_t> cross_targets = {a, one, plus};

  uint64_t hash_key = 956888297470;
  auto hash_output = hash_key;
  for (int i = 0; i < cross_targets.size(); ++i) {
    hash_output = FingerprintCat64(hash_output, cross_targets[i]);
  }
  std::cout << hash_output % 100 << std::endl;
  return 0;
}
