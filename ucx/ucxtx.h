#pragma once

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

class UcxTransport {
 public:
  UcxTransport();
  ~UcxTransport();

  bool listen(const std::string& bind_addr);

  bool ensure_connected(const std::string& peer_addr);

  bool send(const std::string& peer_addr, const void* data, std::size_t size);
  bool recv(const std::string& peer_addr, void* data, std::size_t size);

  bool send(const std::string& peer_addr, const std::vector<std::uint8_t>& buf);
  bool recv(const std::string& peer_addr, std::vector<std::uint8_t>& buf);

  std::vector<std::string> list_peers() const;

 private:
  struct Endpoint {
    ucp_ep_h ep{nullptr};
  };

  ucp_context_h context_{nullptr};
  ucp_worker_h worker_{nullptr};
  ucp_listener_h listener_{nullptr};

  mutable std::mutex mutex_;
  std::unordered_map<std::string, std::shared_ptr<Endpoint>> eps_;
  std::atomic<std::uint64_t> peer_id_counter_{0};

  static constexpr ucp_tag_t SIZE_TAG = 0x3001;
  static constexpr ucp_tag_t DATA_TAG = 0x3002;
  static constexpr ucp_tag_t TAG_MASK = UINT64_MAX;

  struct RequestContext {
    bool completed;
    ucs_status_t status;
  };

  static void request_init(void* req);
  static void send_cb(void* request, ucs_status_t status);
  static void recv_cb(void* request, ucs_status_t status, ucp_tag_recv_info_t* info);

  void init_ucx();
  void close_ep(ucp_ep_h ep);
  void cleanup_ucx();

  std::shared_ptr<Endpoint> get_or_create_ep(const std::string& peer_addr);
  std::shared_ptr<Endpoint> get_or_create_ep_locked(const std::string& peer_addr);

  static bool parse_ip_port(const std::string& s, std::string& ip, uint16_t& port);

  std::shared_ptr<Endpoint> create_client_ep(const std::string& peer_addr);

  static void listener_conn_cb(ucp_conn_request_h conn_req, void* arg) {
    auto* self = reinterpret_cast<UcxTransport*>(arg);
    self->on_conn_request(conn_req);
  }

  void on_conn_request(ucp_conn_request_h conn_req);

  bool build_peer_key_from_conn_req(ucp_conn_request_h conn_req, std::string& out_key);

  bool tag_send_blocking(ucp_ep_h ep, const void* buf, std::size_t size, ucp_tag_t tag);
  bool tag_recv_blocking(void* buf, std::size_t size, ucp_tag_t tag);
  bool wait_request(RequestContext* req, const char* op);
};
