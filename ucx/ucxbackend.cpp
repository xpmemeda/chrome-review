#include "ucxbackend.h"

#include <stdio.h>
#include <ucp/api/ucp.h>
#include <cstring>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>

void ep_error_callback(void* arg, ucp_ep_h ep, ucs_status_t status) { return; }

UcxBackend::UcxBackend(const std::string& address)
    : dist_store_(address), address_(address), context_(nullptr), worker_(nullptr), memh_(nullptr) {}

UcxBackend::~UcxBackend() {
  for (auto& it : ucp_endpoints_) {
    if (it.second.e != nullptr) {
      SafeCloseEndpoint(it.second.e);
    }
  }
  if (memh_ != nullptr) {
    ucp_mem_unmap(context_, memh_);
  }
  if (worker_ != nullptr) {
    ucp_worker_destroy(worker_);
  }
  if (context_ != nullptr) {
    ucp_cleanup(context_);
  }
}

void UcxBackend::Init() {
  ucp_params_t params = {};
  params.field_mask = UCP_PARAM_FIELD_FEATURES;
  params.features = UCP_FEATURE_TAG | UCP_FEATURE_RMA;

  ucp_config_t* config;
  ucp_config_read(NULL, NULL, &config);

  auto status = ucp_init(&params, config, &context_);
  if (status != UCS_OK) {
    LogError("ucp_init fail, err: {}", ucs_status_string(status));
    throw("upc_init fail");
  }

  ucp_config_release(config);

  ucp_worker_params_t worker_params = {};
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
  status = ucp_worker_create(context_, &worker_params, &worker_);
  if (status != UCS_OK) {
    LogError("ucp_worker_create fail, err: {}", ucs_status_string(status));
    throw("ucp_worker_create fail");
  }

  ucp_address_t* local_address = nullptr;
  size_t address_size = 0;
  status = ucp_worker_get_address(worker_, &local_address, &address_size);
  if (status != UCS_OK) {
    LogError("ucp_worker_get_address fail, err: {}", ucs_status_string(status));
    throw("ucp_worker_get_address fail");
  }

  ucx_address_ = std::string((char*)local_address, address_size);
  LogDebug("address_size {} address_hex {}", address_size, comm::BinaryToHex(ucx_address_).c_str());
}

void UcxBackend::RegisterBuffer(void* buffer, int64_t size) {
  ucp_mem_map_params_t params = {};
  params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
  params.address = buffer;
  params.length = size;
  auto status = ucp_mem_map(context_, &params, &memh_);
  if (status != UCS_OK) {
    LogError("ucp_mem_map fail, err: {}", ucs_status_string(status));
    throw("ucp_mem_map fail");
  }
}

comm::RetCode UcxBackend::SendBytes(void* buffer, int64_t size, uint64_t tag, const std::string& dst_address) {
  if (dst_address.size() == 0) {
    LogError("invalid dst_address, size {}", dst_address.size());
    return comm::RetCode::ARG_FAIL;
  }

  comm::IntervalTiming interval;
  UcxEndpoint* ep = GetUcpEndpoint(dst_address);
  if (ep == nullptr) {
    return comm::RetCode::SYS_FAIL;
  }

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH, .memh = memh_};

  LogDebug("ucx start to send, size {} tag {}", size, tag);

  ucs_status_ptr_t status = ucp_tag_send_nbx(ep->e, buffer, size, tag, &param);

  auto ret = CheckStatus(status, UCX_TRANSFER_TIMEOUT_MS);
  if (ret != comm::RetCode::OK) {
    if (ret == comm::RetCode::TIMEOUT) {
      LogError("timeout, size {}", size);
      return ret;
    } else {
      LogError("fail, size {}", size);
      return ret;
    }
  }
  LogHead("ucxSend ok, use {} buffer_size {}, use_time {}ms", ep->protocol, size, interval.PassTime());
  return ret;
}

comm::RetCode UcxBackend::RecvBytes(void* buffer, int64_t size, uint64_t tag, const std::string& src_address) {
  if (src_address.size() == 0) {
    LogError("invalid src_address, size {}", src_address.size());
    return comm::RetCode::ARG_FAIL;
  }

  comm::IntervalTiming interval;
  LogDebug("ucx start to recv, size {} tag {}", size, tag);

  UcxEndpoint* ep = GetUcpEndpoint(src_address);
  if (ep == nullptr) {
    return comm::RetCode::SYS_FAIL;
  }

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH, .memh = memh_};

  ucs_status_ptr_t status = ucp_tag_recv_nbx(worker_, buffer, size, tag, ~0ULL, &param);

  auto ret = CheckStatus(status, UCX_TRANSFER_TIMEOUT_MS);
  if (ret != comm::RetCode::OK) {
    if (ret == comm::RetCode::TIMEOUT) {
      LogError("timeout, size {}", size);
      return ret;
    } else {
      LogError("fail, size {}", size);
      return ret;
    }
  }

  LogHead("ucxRecv ok, use {} buffer_size {}, use_time {}ms", ep->protocol, size, interval.PassTime());
  return ret;
}

UcxEndpoint* UcxBackend::GetUcpEndpoint(const std::string& address) {
  auto it = ucp_endpoints_.find(address);
  if (it != ucp_endpoints_.end()) {
    it->second.last_active_time = comm::TimestampApi::NowSecond();
    return &it->second;
  }

  std::string hex_ucx_address = comm::BinaryToHex(ucx_address_);
  while (dist_store_.set(address, address_, hex_ucx_address) != comm::RetCode::OK) {
    ;
  }
  LogInfo("set address {} succ", address);
  while (!dist_store_.check(address)) {
    ;
  }

  std::string ucx_address = comm::HexToBinary(dist_store_.get(address));

  size_t ucp_address_size = ucx_address.size();
  ucp_address_t* ucp_address = (ucp_address_t*)malloc(ucp_address_size);
  std::memcpy((char*)ucp_address, ucx_address.c_str(), ucp_address_size);
  ucp_ep_params_t ep_params = {.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                             UCP_EP_PARAM_FIELD_ERR_HANDLER,
      .address = ucp_address,
      .err_mode = UCP_ERR_HANDLING_MODE_PEER,
      .err_handler = {.cb = ep_error_callback, .arg = nullptr}};
  ucp_ep_h ucp_endpoint;
  auto status = ucp_ep_create(worker_, &ep_params, &ucp_endpoint);
  if (status != UCS_OK) {
    LogError("ucp_ep_create fail, address {} ucx_address_size {} err: {}", address.c_str(), ucx_address.size(),
        ucs_status_string(status));
    return nullptr;
  }

  UcxEndpoint ucx_e;
  ucx_e.e = ucp_endpoint;
  ucx_e.address = address;
  ucx_e.last_active_time = comm::TimestampApi::NowSecond();
  ucx_e.protocol = GetEpTransport(ucp_endpoint);
  ucp_endpoints_.insert(std::make_pair(address, ucx_e));
  LogHead("new endpoint ok, address {}", address.c_str());
  return &ucp_endpoints_[address];
}

std::string UcxBackend::GetEpTransport(ucp_ep_h ep) {
  if (!ep) return "";

  char* buf = nullptr;
  size_t size = 0;
  FILE* stream = open_memstream(&buf, &size);
  if (!stream) return "";

  ucp_ep_print_info(ep, stream);
  fflush(stream);
  fclose(stream);

  std::string output(buf, size);
  free(buf);

  std::regex lane_regex(R"(lane\[\d+\]:\s+(\S+))");
  std::smatch match;
  std::set<std::string> transports;

  std::string::const_iterator searchStart(output.cbegin());
  while (std::regex_search(searchStart, output.cend(), match, lane_regex)) {
    std::string lane_info = match[1].str();
    auto pos = lane_info.find('/');
    std::string transport = (pos != std::string::npos) ? lane_info.substr(0, pos) : lane_info;
    transports.insert(transport);
    searchStart = match.suffix().first;
  }

  std::ostringstream oss;
  bool first = true;
  for (const auto& t : transports) {
    if (!first) oss << ",";
    oss << t;
    first = false;
  }
  return oss.str();
}

void UcxBackend::CleanExpiredEndpoint() {
  LogDebug("all endpoint size {}", ucp_endpoints_.size());
  std::vector<UcxEndpoint> to_removes;
  auto now_time = comm::TimestampApi::NowSecond();
  for (auto& it : ucp_endpoints_) {
    auto& ucx_e = it.second;
    if (now_time - ucx_e.last_active_time > UCX_ENDPOINT_EXPIRE_TIME_SECOND) {
      LogError("ucx address {} endpoint timeout, last_active_time {} now_time {}", ucx_e.address.c_str(),
          ucx_e.last_active_time, now_time);
      to_removes.push_back(ucx_e);
    }
  }

  for (auto& ucx_e : to_removes) {
    RemoveEndpoint(ucx_e);
  }
}

void UcxBackend::RemoveEndpointByAddress(const std::string& address) {
  auto it = ucp_endpoints_.find(address);
  if (it == ucp_endpoints_.end()) {
    return;
  }

  RemoveEndpoint(it->second);
}

void UcxBackend::RemoveEndpoint(const UcxEndpoint& ucx_e) {
  LogDebug("start remove ucx endpoint, address {}", ucx_e.address.c_str());
  SafeCloseEndpoint(ucx_e.e);
  ucp_endpoints_.erase(ucx_e.address);
  LogDebug("remove ucx endpoint ok, address {}", ucx_e.address.c_str());
}

void UcxBackend::SafeCloseEndpoint(ucp_ep_h ep) {
  if (ep == nullptr) {
    return;
  }

  ucp_request_param_t param = {.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS, .flags = UCP_EP_CLOSE_FLAG_FORCE};

  auto status = ucp_ep_close_nbx(ep, &param);
  if (status == nullptr) {
    LogDebug("close endpoint immediately ok");
    return;
  }

  if (UCS_PTR_IS_ERR(status)) {
    LogError("ucp_ep_close_nbx fail");
    return;
  }

  auto start_time_ms = comm::TimestampApi::NowMilliSecond();
  while (ucp_request_check_status(status) == UCS_INPROGRESS) {
    ucp_worker_progress(worker_);
    comm::TimestampApi::MilliSleep(1);
    auto now_time_ms = comm::TimestampApi::NowMilliSecond();
    if (now_time_ms - start_time_ms > UCX_CLOSE_ENDPOINT_TIMEOUT_MS) {
      LogError("close endpoint timeout, start_time {} now_time {}", start_time_ms, now_time_ms);
      break;
    }
  }
  ucp_request_free(status);
}

uint64_t UcxBackend::MakeTag(const std::string& task_id, int version) {
  auto key = comm::StringPrintf("%s_%d", task_id.c_str(), version);
  return comm::SimpleHashString(key);
}

comm::RetCode UcxBackend::CheckStatus(ucs_status_ptr_t status, int timeout_ms) {
  if (UCS_PTR_IS_ERR(status)) {
    return comm::RetCode::SYS_FAIL;
  }
  if (status == nullptr) {
    return comm::RetCode::OK;
  }

  auto start_time = comm::TimestampApi::NowMilliSecond();
  ucs_status_t s;
  int check_status_times = 0;
  bool is_timeout = false;
  do {
    check_status_times++;
    ucp_worker_progress(worker_);
    s = ucp_request_check_status(status);

    if (check_status_times % 3000 == 0) {
      int pass_time = (int)(comm::TimestampApi::NowMilliSecond() - start_time);
      if (!is_timeout) {
        if (pass_time >= timeout_ms) {
          ucp_request_cancel(worker_, status);
          is_timeout = true;
        }
      } else {
        if (pass_time >= timeout_ms * 1.1) {
          LogError(
              "ucp_request_cancel not working, exiting immediately, pass_time {} timeout_ms {}", pass_time, timeout_ms);
          ucp_request_free(status);
          return comm::RetCode::TIMEOUT;
        }
      }
      comm::TimestampApi::MilliSleep(1);
    }
  } while (s == UCS_INPROGRESS);

  ucp_request_free(status);

  if (is_timeout) {
    return comm::RetCode::TIMEOUT;
  }
  if (s != UCS_OK) {
    return comm::RetCode::SYS_FAIL;
  }

  return comm::RetCode::OK;
}
