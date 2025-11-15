#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <queue>
#include <shared_mutex>
#include <string>
#include <thread>

#include "comm/def.h"

namespace comm {

struct KeyValue {
  std::string key;
  std::string value;
  int op;

  std::string ToJsonString() const;
  RetCode FromJsonString(const std::string& json_str);
};

struct EndpointHealthyStatus {
  int64_t last_active_time_ms;
  bool is_healthy;
};

enum class DistStoreOp { Unknown = 0, Broadcast = 1 };

#define DISTSTORE_TIMEOUT 20
typedef std::function<void(const KeyValue& kv)> DistStoreCallback;

class DistStore {
 public:
  DistStore(
      const std::string& endpoint, const std::string& name = "diststore", const int timeout_ms = DISTSTORE_TIMEOUT);
  DistStore(const std::string& endpoint, const std::vector<std::string>& broadcast_endpoints,
      const std::string& name = "diststore", const int timeout_ms = DISTSTORE_TIMEOUT);
  ~DistStore();

  RetCode set(
      const std::string& endpoint, const std::string& key, const std::string& value, bool need_broadcast = false);
  bool check(const std::string& key);
  std::string get(const std::string& key);

  void RegisterCallback(DistStoreCallback callback);

 private:
  void WorkerLoop();
  RetCode Send(const std::string& endpoint, const std::string& data);
  void Dispatch(const std::string& data);

  void BroadcastAdd(const KeyValue& kv);
  void BroadcastWorkerLoop();

  void CleanWorkerLoop();

 private:
  void HealthyCheckWorkerLoop();
  bool IsEndpointHealthy(const std::string& endpoint);
  void SetEndpointHealthyStatus(const std::string& endpoint, bool is_healthy);

 private:
  std::atomic<bool> stop_workers_;

  std::thread worker_;
  std::thread broadcast_worker_;
  std::string endpoint_;
  std::vector<std::string> broadcast_endpoints_;
  std::string name_;
  int timeout_ms_;
  std::shared_mutex m_;
  std::map<std::string, std::string> kv_;

  std::queue<std::shared_ptr<KeyValue>> broadcast_queue_;
  std::mutex broadcast_queue_m_;

  std::thread clean_worker_;
  std::mutex clean_queue_m_;
  std::queue<std::pair<std::string, uint64_t>> clean_queue_;

  bool has_callback_;
  DistStoreCallback callback_;

  std::thread healthy_check_worker_;
  std::mutex healthy_check_m_;
  std::map<std::string, EndpointHealthyStatus> healthys_;
};

}  // namespace comm