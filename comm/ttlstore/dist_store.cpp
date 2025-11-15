#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <zmq.hpp>

#include "comm/log/log.h"
#include "comm/string/string.h"
#include "comm/time/time.h"
#include "dist_store.h"

namespace comm {

#define TTL_DISTSTORE_EXPIRE_TIME_MS 180000
#define TTL_DISTSTORE_HEALTHY_CHECK_KEY "__dist_store_healthy_check_key__"
#define TTL_DISTSTORE_ACTIVE_ENDPOINT_EXPIRE_TIME_MS 600000

std::string KeyValue::ToJsonString() const {
  nlohmann::json j;
  j["key"] = key;
  j["value"] = value;
  j["op"] = op;
  return j.dump();
}

RetCode KeyValue::FromJsonString(const std::string& json_str) {
  try {
    nlohmann::json j = nlohmann::json::parse(json_str);
    key = j.value("key", "");
    value = j.value("value", "");
    op = j.value("op", 0);
    return RetCode::OK;
  } catch (const std::exception& e) {
    LogError("Failed to parse KeyValue from JSON: json {} e {}", json_str, e.what());
    return RetCode::ARG_FAIL;
  }
}

DistStore::DistStore(const std::string& endpoint, const std::string& name, const int timeout_ms)
    : stop_workers_(false), endpoint_(endpoint), name_(name), timeout_ms_(timeout_ms) {
  has_callback_ = false;
  worker_ = std::thread(&DistStore::WorkerLoop, this);
  clean_worker_ = std::thread(&DistStore::CleanWorkerLoop, this);
  healthy_check_worker_ = std::thread(&DistStore::HealthyCheckWorkerLoop, this);
}

DistStore::DistStore(const std::string& endpoint, const std::vector<std::string>& broadcast_endpoints,
    const std::string& name, const int timeout_ms)
    : endpoint_(endpoint), broadcast_endpoints_(broadcast_endpoints), name_(name), timeout_ms_(timeout_ms) {
  has_callback_ = false;
  worker_ = std::thread(&DistStore::WorkerLoop, this);
  broadcast_worker_ = std::thread(&DistStore::BroadcastWorkerLoop, this);
  clean_worker_ = std::thread(&DistStore::CleanWorkerLoop, this);
  healthy_check_worker_ = std::thread(&DistStore::HealthyCheckWorkerLoop, this);
}

DistStore::~DistStore() {
  stop_workers_.store(true);
  worker_.join();
  clean_worker_.join();
  healthy_check_worker_.join();
}

RetCode DistStore::set(
    const std::string& endpoint, const std::string& key, const std::string& value, bool need_broadcast) {
  if (!IsEndpointHealthy(endpoint)) {
    LogError("endpoint {} not healthy, fast reject", endpoint.c_str());
    return RetCode::FAST_REJECT;
  }

  comm::KeyValue kv;
  kv.key = key;
  kv.value = value;
  if (need_broadcast) {
    kv.op = (int)DistStoreOp::Broadcast;
  }
  auto json_str = kv.ToJsonString();
  auto ret = Send(endpoint, json_str);
  if (ret != RetCode::OK) {
    SetEndpointHealthyStatus(endpoint, false);
  }
  return ret;
}

bool DistStore::check(const std::string& key) {
  std::shared_lock lock(m_);
  return kv_.find(key) != kv_.end();
}

std::string DistStore::get(const std::string& key) {
  std::shared_lock lock(m_);
  return kv_[key];
}

void DistStore::RegisterCallback(DistStoreCallback callback) {
  has_callback_ = true;
  callback_ = callback;
}

void DistStore::WorkerLoop() {
  LogHead("start, endpoint {}", endpoint_.c_str());

  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REP);
  socket.bind(comm::StringPrintf("tcp://%s", endpoint_.c_str()));
  socket.set(zmq::sockopt::sndtimeo, timeout_ms_);

  while (!stop_workers_) {
    zmq::message_t request;
    auto size = socket.recv(request, zmq::recv_flags::none);
    if (size <= 0) {
      LogError("recv req fail");
      continue;
    }

    std::string data(static_cast<char*>(request.data()), request.size());
    Dispatch(data);

    zmq::message_t reply(1);
    size = socket.send(reply, zmq::send_flags::none);
    if (size <= 0) {
      LogError("send resp fail");
    }
  }

  socket.close();
}

RetCode DistStore::Send(const std::string& endpoint, const std::string& data) {
  zmq::context_t context(1);
  zmq::socket_t socket(context, ZMQ_REQ);
  socket.set(zmq::sockopt::linger, 0);
  socket.set(zmq::sockopt::sndtimeo, timeout_ms_);
  socket.set(zmq::sockopt::rcvtimeo, timeout_ms_);
  socket.connect(comm::StringPrintf("tcp://%s", endpoint.c_str()));

  zmq::message_t request(data.begin(), data.end());
  auto size = socket.send(request, zmq::send_flags::none);
  if (size != data.size()) {
    LogError("send fail, endpoint {} data_size {} send_size {}", endpoint.c_str(), data.size(), *size);
    socket.close();
    return RetCode::SYS_FAIL;
  }

  zmq::message_t reply;
  size = socket.recv(reply, zmq::recv_flags::none);
  if (size <= 0) {
    LogError("recv resp fail, endpoint {}", endpoint.c_str());
    socket.close();
    return RetCode::SYS_FAIL;
  }

  return RetCode::OK;
}

void DistStore::Dispatch(const std::string& data) {
  comm::KeyValue kv;
  auto ret = kv.FromJsonString(data);
  if (ret != RetCode::OK) {
    return;
  }

  if (kv.key == TTL_DISTSTORE_HEALTHY_CHECK_KEY) {
    return;
  }

  if (has_callback_) {
    callback_(kv);
  } else {
    std::unique_lock lock(m_);
    kv_[kv.key] = kv.value;

    std::unique_lock queue_lock(clean_queue_m_);
    clean_queue_.push(std::make_pair(kv.key, TimestampApi::NowMilliSecond()));
  }

  if (kv.op == int(DistStoreOp::Broadcast)) {
    BroadcastAdd(kv);
  }
}

void DistStore::BroadcastAdd(const KeyValue& kv) {
  if (broadcast_endpoints_.size() <= 1) {
    return;
  }

  auto kv_ptr = std::make_shared<KeyValue>(kv);
  std::unique_lock lock(broadcast_queue_m_);
  broadcast_queue_.push(kv_ptr);
}

void DistStore::BroadcastWorkerLoop() {
  LogDebug("start");
  while (true) {
    std::unique_lock lock(broadcast_queue_m_);
    if (!broadcast_queue_.empty()) {
      auto kv_ptr = broadcast_queue_.front();
      lock.unlock();

      bool done = true;
      for (auto& endpoint : broadcast_endpoints_) {
        if (endpoint != endpoint_) {
          auto ret = set(endpoint, kv_ptr->key, kv_ptr->value, false);
          if (ret != RetCode::OK) {
            done = false;
            break;
          }
        }
      }

      if (done) {
        std::unique_lock lock(broadcast_queue_m_);
        broadcast_queue_.pop();
      }
    } else {
      lock.unlock();
      TimestampApi::MilliSleep(1);
    }
  }
}

void DistStore::CleanWorkerLoop() {
  while (!stop_workers_) {
    std::vector<std::string> expried_keys;
    std::unique_lock lock(clean_queue_m_);
    if (!clean_queue_.empty()) {
      auto now_time = TimestampApi::NowMilliSecond();
      while (!clean_queue_.empty()) {
        auto& front = clean_queue_.front();
        if (now_time - front.second < TTL_DISTSTORE_EXPIRE_TIME_MS) {
          break;
        }
        expried_keys.push_back(front.first);
        clean_queue_.pop();
      }
      lock.unlock();

      {
        std::unique_lock lock(m_);
        for (auto& key : expried_keys) {
          kv_.erase(key);
        }
      }
    } else {
      lock.unlock();
    }
    TimestampApi::MilliSleep(1000);
  }
}

void DistStore::HealthyCheckWorkerLoop() {
  while (!stop_workers_) {
    auto now_time = TimestampApi::NowMilliSecond();
    std::unique_lock lock(healthy_check_m_);
    std::vector<std::string> not_healthy_endpoints;
    std::vector<std::string> expired_endpoints;
    for (auto& it : healthys_) {
      if (now_time - it.second.last_active_time_ms > TTL_DISTSTORE_ACTIVE_ENDPOINT_EXPIRE_TIME_MS) {
        LogDebug("endpoint {} not active, remove from healthy check", it.first.c_str());
        expired_endpoints.push_back(it.first);
      } else if (!it.second.is_healthy) {
        not_healthy_endpoints.push_back(it.first);
      }
    }
    for (auto& e : expired_endpoints) {
      healthys_.erase(e);
    }
    lock.unlock();

    comm::KeyValue kv;
    kv.key = TTL_DISTSTORE_HEALTHY_CHECK_KEY;
    kv.value = "check";
    auto json_str = kv.ToJsonString();

    for (auto& e : not_healthy_endpoints) {
      auto ret = Send(e, json_str);
      if (ret == RetCode::OK) {
        SetEndpointHealthyStatus(e, true);
      }
    }
    TimestampApi::MilliSleep(500);
  }
}

bool DistStore::IsEndpointHealthy(const std::string& endpoint) {
  std::unique_lock lock(healthy_check_m_);
  auto it = healthys_.find(endpoint);
  if (it != healthys_.end()) {
    it->second.last_active_time_ms = TimestampApi::NowMilliSecond();
    return it->second.is_healthy;
  }
  EndpointHealthyStatus status;
  status.last_active_time_ms = TimestampApi::NowMilliSecond();
  status.is_healthy = true;
  healthys_.insert(std::make_pair(endpoint, status));
  return true;
}

void DistStore::SetEndpointHealthyStatus(const std::string& endpoint, bool is_healthy) {
  if (!is_healthy) {
    LogError("set endpoint {} not healthy", endpoint.c_str());
  } else {
    LogHead("set endpoint {} healthy", endpoint.c_str());
  }
  std::unique_lock lock(healthy_check_m_);
  auto it = healthys_.find(endpoint);
  if (it != healthys_.end()) {
    it->second.is_healthy = is_healthy;
  }
}

}  // namespace comm
