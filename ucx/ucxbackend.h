#pragma once

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <ucp/api/ucp.h>
#include <map>
#include <set>
#include <string>
#include <vector>
#include "comm/comm.h"

DEFINE_ENV_VAR(UCX_TRANSFER_TIMEOUT_MS, 5000);
DEFINE_ENV_VAR(UCX_CLOSE_ENDPOINT_TIMEOUT_MS, 500);
DEFINE_ENV_VAR(UCX_ENDPOINT_EXPIRE_TIME_SECOND, 3600 * 24);

struct UcxEndpoint {
  ucp_ep_h e;
  std::string address;
  uint32_t last_active_time;
  std::string protocol;
};

class UcxBackend {
 public:
  UcxBackend(const std::string& address);
  ~UcxBackend();

  void Init();
  void RegisterBuffer(void* buffer, int64_t size);

  comm::RetCode SendBytes(void* buffer, int64_t size, uint64_t tag, const std::string& dst_address);
  comm::RetCode RecvBytes(void* buffer, int64_t size, uint64_t tag, const std::string& src_address);

  uint64_t MakeTag(const std::string& task_id, int version);

  std::string& get_address() { return ucx_address_; }

 private:
  UcxEndpoint* GetUcpEndpoint(const std::string& address);
  std::string GetEpTransport(ucp_ep_h ep);
  comm::RetCode CheckStatus(ucs_status_ptr_t status, int timeout_ms);

  void CleanExpiredEndpoint();

  void RemoveEndpointByAddress(const std::string& address);
  void RemoveEndpoint(const UcxEndpoint& ucx_e);
  void SafeCloseEndpoint(ucp_ep_h ep);

 private:
  comm::DistStore dist_store_;

  std::string address_;
  std::string ucx_address_;

  ucp_context_h context_;
  ucp_worker_h worker_;
  ucp_mem_h memh_;
  std::map<std::string, UcxEndpoint> ucp_endpoints_;
};
