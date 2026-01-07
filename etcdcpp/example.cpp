#include <algorithm>
#include <chrono>
#include <etcd/Client.hpp>
#include <etcd/Response.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

const std::string PREFIX = "/demo_kvcache";

std::string owner_key(const std::string& tag, const std::string& owner_id) {
  return PREFIX + "/tag/" + tag + "/owners/" + owner_id;
}

std::string owners_prefix(const std::string& tag) { return PREFIX + "/tag/" + tag + "/owners/"; }

// publish: set owner under each tag with a lease
void publish(etcd::Client& etcd, const std::vector<std::string>& tags, const std::string& owner_id, int64_t lease_id) {
  for (const auto& t : tags) {
    etcd::Response resp = etcd.set(owner_key(t, owner_id), "1", lease_id).get();
    if (!resp.is_ok()) {
      std::cerr << "etcd.set err code " << resp.error_code() << std::endl;
    }
  }
}

// unpublish: delete owner under each tag
void unpublish(etcd::Client& etcd, const std::vector<std::string>& tags, const std::string& owner_id) {
  for (const auto& t : tags) {
    etcd.rm(owner_key(t, owner_id)).wait();
  }
}

// pick any owner under a tag
std::optional<std::string> pick_any_owner(etcd::Client& etcd, const std::string& tag) {
  etcd::Response resp = etcd.ls(owners_prefix(tag)).get();
  if (!resp.is_ok() || resp.keys().empty()) {
    return std::nullopt;
  }

  auto key = resp.key(0);  // pick the first key
  std::string full_key = key;
  auto pos = full_key.find_last_of('/');
  if (pos == std::string::npos) return std::nullopt;
  return full_key.substr(pos + 1);
}

// get last hit: from back to front
std::pair<std::optional<std::string>, int> get_last_hit(etcd::Client& etcd, const std::vector<std::string>& tags) {
  for (int i = static_cast<int>(tags.size()) - 1; i >= 0; --i) {
    auto owner = pick_any_owner(etcd, tags[i]);
    if (owner.has_value()) {
      return {owner, i + 1};
    }
  }
  return {std::nullopt, 0};
}

// list owners (for debugging)
std::vector<std::string> list_owners(etcd::Client& etcd, const std::string& tag) {
  std::vector<std::string> res;
  etcd::Response resp = etcd.ls(owners_prefix(tag)).get();
  if (!resp.is_ok()) {
    std::cerr << "etcd.ls err code " << resp.error_code() << std::endl;
  }

  for (size_t i = 0; i < resp.keys().size(); ++i) {
    std::string full_key = resp.key(i);
    auto pos = full_key.find_last_of('/');
    if (pos != std::string::npos) {
      res.push_back(full_key.substr(pos + 1));
    }
  }
  std::sort(res.begin(), res.end());
  return res;
}

// cleanup prefix
void cleanup_prefix(etcd::Client& etcd) {
  etcd::Response resp = etcd.get(PREFIX + "/").get();
  for (size_t i = 0; i < resp.keys().size(); ++i) {
    etcd.rm(resp.key(i)).wait();
  }
}

int main() {
  etcd::Client etcd("http://127.0.0.1:2379");

  cleanup_prefix(etcd);

  std::string ownerA = "10.0.0.1:9000#epochA";
  std::string ownerB = "10.0.0.2:9000#epochB";

  std::vector<std::string> tagsA{"a", "b", "c"};
  std::vector<std::string> tagsB{"a", "b"};
  std::vector<std::string> query{"a", "b", "c", "d"};

  // create leases
  int64_t leaseA = etcd.leasegrant(5).get().value().lease();   // TTL 5s
  int64_t leaseB = etcd.leasegrant(30).get().value().lease();  // TTL 30s

  publish(etcd, tagsA, ownerA, leaseA);
  publish(etcd, tagsB, ownerB, leaseB);

  std::cout << "== After publish ==" << std::endl;
  for (auto t : {"a", "b", "c"}) {
    auto owners = list_owners(etcd, t);
    std::cout << "owners(" << t << ") = ";
    for (auto& o : owners) std::cout << o << " ";
    std::cout << std::endl;
  }

  auto [owner, n] = get_last_hit(etcd, query);
  std::cout << "get_last_hit [a, b, c, d] => owner=" << (owner.value_or("None")) << ", num_tags=" << n << std::endl;

  std::cout << "\n== Sleep 6s to let leaseA expire ==" << std::endl;
  std::this_thread::sleep_for(6s);

  std::cout << "== After leaseA expired ==" << std::endl;
  for (auto t : {"a", "b", "c"}) {
    auto owners = list_owners(etcd, t);
    std::cout << "owners(" << t << ") = ";
    for (auto& o : owners) std::cout << o << " ";
    std::cout << std::endl;
  }

  auto [owner2, n2] = get_last_hit(etcd, query);
  std::cout << "get_last_hit [a, b, c, d] => owner=" << (owner2.value_or("None")) << ", num_tags=" << n2 << std::endl;

  std::cout << "\n== Unpublish ownerB on [a,b] ==" << std::endl;
  unpublish(etcd, {"a", "b"}, ownerB);
  for (auto t : {"a", "b"}) {
    auto owners = list_owners(etcd, t);
    std::cout << "owners(" << t << ") = ";
    for (auto& o : owners) std::cout << o << " ";
    std::cout << std::endl;
  }

  auto [owner3, n3] = get_last_hit(etcd, query);
  std::cout << "get_last_hit [a, b, c, d] => owner=" << (owner3.value_or("None")) << ", num_tags=" << n3 << std::endl;

  return 0;
}
