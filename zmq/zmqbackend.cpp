#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "comm/comm.h"
#include "zmqbackend.h"

ZmqBackend::ZmqBackend(const std::string& address)
    : address_(comm::StringPrintf("tcp://%s", address.c_str())), ctx_(1), pull_socket_(ctx_, zmq::socket_type::pull) {
  pull_socket_.bind(address_);
}

ZmqBackend::ZmqBackend(const std::string& address, const std::string&) : ZmqBackend(address) {}

ZmqBackend::~ZmqBackend() = default;

std::string ZmqBackend::getLocalMetadata() const { return address_; }

void ZmqBackend::registerBuffer(void* /*buffer*/, size_t /*size*/) {}

ZmqBackend::Endpoint* ZmqBackend::getPeerEndpoint(const std::string& address) {
  auto peer_address = comm::StringPrintf("tcp://%s", address.c_str());

  auto it = endpoints_.find(peer_address);
  if (it != endpoints_.end()) {
    return &it->second;
  }

  Endpoint ep;
  ep.address = peer_address;
  ep.push_socket = std::make_unique<zmq::socket_t>(ctx_, zmq::socket_type::push);
  ep.push_socket->connect(peer_address);

  auto [iter, ok] = endpoints_.emplace(peer_address, std::move(ep));
  (void)ok;
  return &iter->second;
}

bool ZmqBackend::send(void* buffer, size_t size, const std::string& tag, const std::string& peer_address) {
  Endpoint* endpoint = getPeerEndpoint(peer_address);
  if (!endpoint || !endpoint->push_socket) return false;

  zmq::socket_t& sock = *endpoint->push_socket;

  try {
    sock.send(zmq::buffer(tag), zmq::send_flags::sndmore);
    sock.send(zmq::buffer(buffer, size), zmq::send_flags::none);
  } catch (const zmq::error_t& e) {
    LogError("ZmqBackend send error: {}", e.what());
    return false;
  }

  LogInfo("ZmqBackend send tag {} size {} to {}", tag, size, peer_address);
  return true;
}

bool ZmqBackend::recv(void* buffer, size_t size, const std::string& tag, const std::string& /*peer_address*/) {
  auto it = pending_.find(tag);
  if (it != pending_.end()) {
    const auto& payload = it->second;
    if (payload.size() < size) {
      LogError("ZmqBackend recv: payload too small, expect {}, got {}", size, payload.size());
      return false;
    }
    std::memcpy(buffer, payload.data(), size);
    pending_.erase(it);
    LogInfo("ZmqBackend recv tag {} (from cache) size {}", tag, size);
    return true;
  }

  while (true) {
    zmq::message_t tag_msg;
    zmq::message_t payload_msg;

    try {
      if (!pull_socket_.recv(tag_msg, zmq::recv_flags::none)) {
        return false;
      }
      if (!pull_socket_.recv(payload_msg, zmq::recv_flags::none)) {
        return false;
      }
    } catch (const zmq::error_t& e) {
      LogError("ZmqBackend recv error: {}", e.what());
      return false;
    }

    std::string recv_tag(static_cast<const char*>(tag_msg.data()), tag_msg.size());

    if (recv_tag == tag) {
      if (payload_msg.size() < size) {
        LogError("ZmqBackend recv: payload too small, expect {}, got {}", size, payload_msg.size());
        return false;
      }
      std::memcpy(buffer, payload_msg.data(), size);
      LogInfo("ZmqBackend recv tag {} size {}", tag, size);
      return true;
    } else {
      auto& buf = pending_[recv_tag];
      buf.resize(payload_msg.size());
      std::memcpy(buf.data(), payload_msg.data(), payload_msg.size());
      LogInfo("ZmqBackend cache message tag {} size {}", recv_tag, payload_msg.size());
    }
  }

  return false;
}
