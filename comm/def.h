#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace comm {

template <typename T>
T GetDefineEnvValue(const std::string& var_name, T default_value) {
  auto value = default_value;
  const std::string env_key = "APEX_MID_" + var_name;
  const char* env_value = std::getenv(env_key.c_str());
  if (env_value != nullptr) {
    std::cout << "read_env: " << env_key << "=" << env_value << std::endl;
    try {
      if constexpr (std::is_same_v<T, std::string>) {
        value = std::string(env_value);
      } else if constexpr (std::is_integral_v<T>) {
        value = static_cast<T>(std::stoll(env_value));
      } else if constexpr (std::is_floating_point_v<T>) {
        value = static_cast<T>(std::stod(env_value));
      }
    } catch (...) {
      value = default_value;
    }
  }
  std::cout << "define_val: " << var_name << "=" << value << std::endl;
  return value;
}

#define DEFINE_ENV_VAR(var_name, default_value)                                          \
  inline const auto var_name = [] {                                                      \
    return comm::GetDefineEnvValue<decltype(default_value)>((#var_name), default_value); \
  }()

#define PY_ASSERT(cond, msg) \
  if (!(cond)) throw std::runtime_error(msg)

#define P_ROLE "p"
#define D_ROLE "d"

#define ATTN_MLA "mla"
#define ATTN_MHA "mha"

#define TRANSFER_ENGINE_INNER "inner"
#define TRANSFER_ENGINE_KVXFER "kvxfer"

#define TRANSFER_ENGINE_BACKEND_NCCL "nccl"
#define TRANSFER_ENGINE_BACKEND_UCX "ucx"

#define TRANSFER_COPY_MODE_DIRECT "direct"
#define TRANSFER_COPY_MODE_COPY "copy"

#define WORKER_LOOP_SLEEP_MS 1

enum class RetCode : int {
  OK = 0,
  KEY_NOTFOUND = 1,
  SYS_FAIL = -1,
  ARG_FAIL = -2,
  TIMEOUT = -3,
  FAST_REJECT = -4,
  OUT_OF_MEMORY = -10
};

enum class AsyncResult : int { UN_KNOWN = 0, RUNNING = 1, FAIL = 2, SUCCESS = 3 };

enum class ErrorCode : int {
  OK = 0,
  ARG_ERROR = -2,
  TOPOLOGY_CAL_FAIL = -3,
  REJECT_BY_PEER = -11,
  REJECT_BY_MYSELF = -12,
  REJECT_BY_RETRACTED = -13,
  BOOTSTRAP_TIME_OUT = -30,
  KVCACHE_TRANSFER_FAIL = -50,
  KVCACHE_TRANSFER_TIMEOUT = -51,
  UNKNOWN_ERROR = -10000,
};

}  // namespace comm