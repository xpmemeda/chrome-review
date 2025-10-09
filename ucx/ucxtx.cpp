#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <ucp/api/ucp.h>
#include <unistd.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ucxtx.h"

UcxTransport::UcxTransport() { init_ucx(); }
UcxTransport::~UcxTransport() { cleanup_ucx(); }

bool UcxTransport::listen(const std::string& bind_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (listener_) {
    std::cerr << "listener already created\n";
    return false;
  }

  std::string ip;
  uint16_t port = 0;
  if (!parse_ip_port(bind_addr, ip, port)) {
    std::cerr << "invalid bind addr: " << bind_addr << "\n";
    return false;
  }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (ip.empty() || ip == "0.0.0.0") {
    addr.sin_addr.s_addr = INADDR_ANY;
  } else {
    if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
      std::cerr << "inet_pton failed for " << ip << "\n";
      return false;
    }
  }

  ucp_listener_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR | UCP_LISTENER_PARAM_FIELD_CONN_HANDLER;
  params.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addr);
  params.sockaddr.addrlen = sizeof(addr);
  params.conn_handler.cb = &UcxTransport::listener_conn_cb;
  params.conn_handler.arg = this;

  ucs_status_t st = ucp_listener_create(worker_, &params, &listener_);
  if (st != UCS_OK) {
    std::cerr << "ucp_listener_create failed: " << ucs_status_string(st) << "\n";
    listener_ = nullptr;
    return false;
  }
  return true;
}

bool UcxTransport::ensure_connected(const std::string& peer_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<bool>(get_or_create_ep_locked(peer_addr));
}

bool UcxTransport::send(const std::string& peer_addr, const void* data, std::size_t size) {
  std::shared_ptr<UcxTransport::Endpoint> ep = get_or_create_ep(peer_addr);
  if (!ep || !ep->ep) return false;
  return tag_send_blocking(ep->ep, data, size, DATA_TAG);
}

bool UcxTransport::recv(const std::string& peer_addr, void* data, std::size_t size) {
  std::shared_ptr<UcxTransport::Endpoint> ep = get_or_create_ep(peer_addr);
  if (!ep || !ep->ep) {
    return false;
  }
  return tag_recv_blocking(data, size, DATA_TAG);
}

bool UcxTransport::send(const std::string& peer_addr, const std::vector<std::uint8_t>& buf) {
  std::shared_ptr<UcxTransport::Endpoint> ep = get_or_create_ep(peer_addr);
  if (!ep || !ep->ep) {
    return false;
  }

  std::uint64_t len = buf.size();
  if (!tag_send_blocking(ep->ep, &len, sizeof(len), SIZE_TAG)) {
    return false;
  }
  if (len == 0) {
    return true;
  }
  return tag_send_blocking(ep->ep, buf.data(), buf.size(), DATA_TAG);
}

bool UcxTransport::recv(const std::string& peer_addr, std::vector<std::uint8_t>& buf) {
  std::shared_ptr<UcxTransport::Endpoint> ep = get_or_create_ep(peer_addr);
  if (!ep || !ep->ep) {
    return false;
  }

  std::uint64_t len = 0;
  if (!tag_recv_blocking(&len, sizeof(len), SIZE_TAG)) {
    return false;
  }
  buf.resize(static_cast<std::size_t>(len));
  if (len == 0) {
    return true;
  }
  return tag_recv_blocking(buf.data(), buf.size(), DATA_TAG);
}

std::vector<std::string> UcxTransport::list_peers() const {
  ucp_worker_progress(worker_);

  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<std::string> res;
  res.reserve(eps_.size());
  for (auto& kv : eps_) {
    res.push_back(kv.first);
  }
  return res;
}

void UcxTransport::request_init(void* req) {
  auto* ctx = reinterpret_cast<RequestContext*>(req);
  ctx->completed = false;
  ctx->status = UCS_INPROGRESS;
}

void UcxTransport::send_cb(void* request, ucs_status_t status) {
  auto* ctx = reinterpret_cast<RequestContext*>(request);
  ctx->status = status;
  ctx->completed = true;
}

void UcxTransport::recv_cb(void* request, ucs_status_t status, ucp_tag_recv_info_t* info) {
  auto* ctx = reinterpret_cast<RequestContext*>(request);
  ctx->status = status;
  ctx->completed = true;
}

void UcxTransport::init_ucx() {
  ucp_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.field_mask = UCP_PARAM_FIELD_FEATURES | UCP_PARAM_FIELD_REQUEST_SIZE | UCP_PARAM_FIELD_REQUEST_INIT;
  params.features = UCP_FEATURE_TAG;
  params.request_size = sizeof(RequestContext);
  params.request_init = request_init;

  ucp_config_t* config = nullptr;
  ucs_status_t st = ucp_config_read(nullptr, nullptr, &config);
  if (st != UCS_OK) {
    throw std::runtime_error("ucp_config_read failed");
  }

  st = ucp_init(&params, config, &context_);
  ucp_config_release(config);
  if (st != UCS_OK) {
    throw std::runtime_error("ucp_init failed");
  }

  ucp_worker_params_t wparams;
  std::memset(&wparams, 0, sizeof(wparams));
  wparams.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  wparams.thread_mode = UCS_THREAD_MODE_SINGLE;

  st = ucp_worker_create(context_, &wparams, &worker_);
  if (st != UCS_OK) {
    ucp_cleanup(context_);
    context_ = nullptr;
    throw std::runtime_error("ucp_worker_create failed");
  }
}

void UcxTransport::close_ep(ucp_ep_h ep) {
  // if (!ep) return;

  // ucp_request_param_t param;
  // std::memset(&param, 0, sizeof(param));
  // // 我们只用 flags 字段：FLUSH 表示把所有 pending 的东西都正常刷完再关
  // param.op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS;
  // param.flags = UCP_EP_CLOSE_FLAG_FLUSH;

  // // 因为在 ucp_init 里已经设置了 request_size / request_init，
  // // 所以这里返回的 request 也是我们的 RequestContext*
  // RequestContext* req = reinterpret_cast<RequestContext*>(ucp_ep_close_nbx(ep, &param));

  // // 使用你现有的 wait_request 等待关闭完成
  // wait_request(req, "ep_close");

  if (!ep) return;

  // 返回值是 request pointer（也就是你自定义 RequestContext）
  RequestContext* req = reinterpret_cast<RequestContext*>(ucp_ep_close_nb(ep, UCP_EP_CLOSE_MODE_FLUSH));

  // ucp_ep_close_nb 可能立即完成（req == NULL），也可能返回一个异步 request
  wait_request(req, "ep_close");
}

void UcxTransport::cleanup_ucx() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (listener_) {
    ucp_listener_destroy(listener_);
    listener_ = nullptr;
  }

  for (auto& kv : eps_) {
    if (kv.second && kv.second->ep) {
      // ucp_ep_destroy(kv.second->ep);
      close_ep(kv.second->ep);
      kv.second->ep = nullptr;
    }
  }
  eps_.clear();

  if (worker_) {
    ucp_worker_destroy(worker_);
    worker_ = nullptr;
  }
  if (context_) {
    ucp_cleanup(context_);
    context_ = nullptr;
  }
}

std::shared_ptr<UcxTransport::Endpoint> UcxTransport::get_or_create_ep(const std::string& peer_addr) {
  std::lock_guard<std::mutex> lock(mutex_);
  return get_or_create_ep_locked(peer_addr);
}

std::shared_ptr<UcxTransport::Endpoint> UcxTransport::get_or_create_ep_locked(const std::string& peer_addr) {
  auto it = eps_.find(peer_addr);
  if (it != eps_.end()) {
    return it->second;
  }

  auto ep = create_client_ep(peer_addr);
  if (!ep) return nullptr;
  eps_[peer_addr] = ep;
  return ep;
}

bool UcxTransport::parse_ip_port(const std::string& s, std::string& ip, uint16_t& port) {
  auto pos = s.find(':');
  if (pos == std::string::npos) return false;
  ip = s.substr(0, pos);
  std::string port_str = s.substr(pos + 1);
  int p = std::stoi(port_str);
  if (p < 0 || p > 65535) return false;
  port = static_cast<uint16_t>(p);
  return true;
}

std::shared_ptr<UcxTransport::Endpoint> UcxTransport::create_client_ep(const std::string& peer_addr) {
  std::string ip;
  uint16_t port = 0;
  if (!parse_ip_port(peer_addr, ip, port)) {
    std::cerr << "Invalid peer addr: " << peer_addr << "\n";
    return nullptr;
  }

  sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (inet_pton(AF_INET, ip.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "inet_pton failed for " << ip << "\n";
    return nullptr;
  }

  ucp_ep_params_t ep_params;
  std::memset(&ep_params, 0, sizeof(ep_params));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_FLAGS | UCP_EP_PARAM_FIELD_SOCK_ADDR;
  ep_params.flags = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER;
  ep_params.sockaddr.addr = reinterpret_cast<const sockaddr*>(&addr);
  ep_params.sockaddr.addrlen = sizeof(addr);

  auto ep = std::make_shared<UcxTransport::Endpoint>();
  ucs_status_t st = ucp_ep_create(worker_, &ep_params, &ep->ep);
  if (st != UCS_OK) {
    std::cerr << "ucp_ep_create(client) failed: " << ucs_status_string(st) << " for peer " << peer_addr << "\n";
    return nullptr;
  }

  for (int i = 0; i < 1000; ++i) {
    ucp_worker_progress(worker_);
  }

  return ep;
}

void UcxTransport::on_conn_request(ucp_conn_request_h conn_req) {
  ucp_ep_params_t ep_params;
  std::memset(&ep_params, 0, sizeof(ep_params));
  ep_params.field_mask = UCP_EP_PARAM_FIELD_CONN_REQUEST;
  ep_params.conn_request = conn_req;

  auto ep = std::make_shared<UcxTransport::Endpoint>();
  ucs_status_t st = ucp_ep_create(worker_, &ep_params, &ep->ep);
  if (st != UCS_OK) {
    std::cerr << "ucp_ep_create(server) failed: " << ucs_status_string(st) << "\n";
    return;
  }

  std::string peer_key;
  if (!build_peer_key_from_conn_req(conn_req, peer_key)) {
    std::uint64_t id = peer_id_counter_.fetch_add(1);
    peer_key = "peer#" + std::to_string(id);
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    eps_[peer_key] = ep;
  }
  std::cerr << "Accepted new connection from " << peer_key << "\n";
}

bool UcxTransport::build_peer_key_from_conn_req(ucp_conn_request_h conn_req, std::string& out_key) {
  ucp_conn_request_attr_t attr;
  memset(&attr, 0, sizeof(attr));
  attr.field_mask = UCP_CONN_REQUEST_ATTR_FIELD_CLIENT_ADDR;

  ucs_status_t st = ucp_conn_request_query(conn_req, &attr);
  if (st != UCS_OK) {
    return false;
  }

  const sockaddr_storage& ss = attr.client_address;
  if (ss.ss_family != AF_INET) {
    return false;
  }

  const sockaddr_in* sin = reinterpret_cast<const sockaddr_in*>(&ss);
  char ipbuf[INET_ADDRSTRLEN] = {0};

  if (!inet_ntop(AF_INET, &sin->sin_addr, ipbuf, sizeof(ipbuf))) {
    return false;
  }

  uint16_t port = ntohs(sin->sin_port);

  out_key = std::string(ipbuf) + ":" + std::to_string(port);
  return true;
}

bool UcxTransport::tag_send_blocking(ucp_ep_h ep, const void* buf, std::size_t size, ucp_tag_t tag) {
  RequestContext* req =
      reinterpret_cast<RequestContext*>(ucp_tag_send_nb(ep, buf, size, ucp_dt_make_contig(1), tag, send_cb));
  return wait_request(req, "send");
}

bool UcxTransport::tag_recv_blocking(void* buf, std::size_t size, ucp_tag_t tag) {
  RequestContext* req = reinterpret_cast<RequestContext*>(
      ucp_tag_recv_nb(worker_, buf, size, ucp_dt_make_contig(1), tag, TAG_MASK, recv_cb));
  return wait_request(req, "recv");
}

bool UcxTransport::wait_request(RequestContext* req, const char* op) {
  if (req == nullptr) {
    return true;
  }
  if (UCS_PTR_IS_ERR(req)) {
    auto st = UCS_PTR_STATUS(req);
    std::cerr << op << " failed: " << ucs_status_string(st) << "\n";
    return false;
  }

  while (!req->completed) {
    ucp_worker_progress(worker_);
  }

  ucs_status_t st = req->status;
  ucp_request_free(req);

  if (st != UCS_OK) {
    std::cerr << op << " completed with error: " << ucs_status_string(st) << "\n";
    return false;
  }
  return true;
}
