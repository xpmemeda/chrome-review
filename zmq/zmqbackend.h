#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <zmq.hpp>

class ZmqBackend {
 public:
  explicit ZmqBackend(const std::string& address);
  ~ZmqBackend();

  ZmqBackend(const std::string& address, const std::string& backend);

  std::string getLocalMetadata() const;

  void registerBuffer(void* buffer, size_t size);

  bool send(void* buffer, size_t size, const std::string& tag, const std::string& peer_address);
  bool recv(void* buffer, size_t size, const std::string& tag, const std::string& peer_address);

 private:
  struct Endpoint {
    std::string address;
    std::unique_ptr<zmq::socket_t> push_socket;
  };

  Endpoint* getPeerEndpoint(const std::string& peer_address);

 private:
  std::string address_;
  zmq::context_t ctx_;
  zmq::socket_t pull_socket_;

  std::unordered_map<std::string, Endpoint> endpoints_;
  std::unordered_map<std::string, std::vector<uint8_t>> pending_;
};
