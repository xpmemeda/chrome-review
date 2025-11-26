#pragma once

#include <nixl.h>
#include <stddef.h>
#include <stdint.h>
#include <memory>
#include <string>

#include "comm/comm.h"

class NixlBackend {
 public:
  struct Endpoint {
    std::string uuid;
  };

 public:
  NixlBackend(const std::string& address, const std::string& backend);
  NixlBackend(const std::string& address);
  ~NixlBackend();

  void registerBuffer(void* buffer, size_t size);
  bool send(void* buffer, size_t size, const std::string& tag, const std::string& peer_address);
  bool recv(void* buffer, size_t size, const std::string& tag, const std::string& peer_address);

 private:
  std::string getLocalMetadata() const;
  Endpoint* getPeerEndpoint(const std::string& peer_address);

 private:
  std::string address_;
  comm::DistStore dist_store_;

  std::string uuid_;
  std::string backend_;
  std::unique_ptr<nixlAgent> agent_;
  nixl_opt_args_t extra_params_;
  nixl_reg_dlist_t reg_buffers_;

  std::unordered_map<std::string, Endpoint> endpoints_;
};
