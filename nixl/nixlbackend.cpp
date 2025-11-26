#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

#include "nixlbackend.h"
#include "utils/serdes/serdes.h"

#define NIXL_CHECK(expr)                                      \
  do {                                                        \
    auto r = expr;                                            \
    if (r != NIXL_SUCCESS) {                                  \
      throw std::runtime_error(#expr + std::string(" err.")); \
    }                                                         \
  } while (0)

std::string load_data_from_file(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) throw std::runtime_error("open file: \"" + path + "\" failed");
  ifs.seekg(0, ifs.end);
  auto length = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> buffer(new char[length]);
  ifs.read(buffer.get(), length);
  std::string data(buffer.get(), length);
  return data;
}

std::string uuid_v4() {
  static thread_local std::random_device rd;
  static thread_local std::mt19937_64 gen(rd());
  static thread_local std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

  uint64_t part1 = dist(gen);
  uint64_t part2 = dist(gen);

  part2 = (part2 & 0x3FFFFFFFFFFFFFFFULL) | 0x8000000000000000ULL;
  part1 = (part1 & 0xFFFFFFFFFFFF0FFFULL) | 0x0000000000004000ULL;

  std::stringstream ss;
  ss << std::hex << std::nouppercase << std::setfill('0') << std::setw(8) << static_cast<uint32_t>(part1 >> 32) << "-"
     << std::setw(4) << static_cast<uint16_t>(part1 >> 16) << "-" << std::setw(4) << static_cast<uint16_t>(part1) << "-"
     << std::setw(4) << static_cast<uint16_t>(part2 >> 48) << "-" << std::setw(12)
     << static_cast<uint64_t>(part2 & 0x0000FFFFFFFFFFFFULL);

  return ss.str();
}

NixlBackend::NixlBackend(const std::string& address, const std::string& backend)
    : address_(address), dist_store_(address), uuid_(uuid_v4()), backend_(backend), reg_buffers_(DRAM_SEG) {
  nixlAgentConfig cfg(true);
  agent_ = std::make_unique<nixlAgent>(uuid_, cfg);

  std::vector<nixl_backend_t> plugins;
  NIXL_CHECK(agent_->getAvailPlugins(plugins));
  if (std::find(plugins.begin(), plugins.end(), backend) == plugins.end()) {
    throw std::runtime_error("unavailable backend: " + backend);
  }

  nixl_b_params_t init_params;
  nixl_mem_list_t mems;
  NIXL_CHECK(agent_->getPluginParams(backend, mems, init_params));
  nixlBackendH* backend_object;
  NIXL_CHECK(agent_->createBackend(backend, init_params, backend_object));
  extra_params_.backends.push_back(backend_object);
}

NixlBackend::NixlBackend(const std::string& address) : NixlBackend(address, "UCX") {}

NixlBackend::~NixlBackend() = default;

std::string NixlBackend::getLocalMetadata() const {
  std::string meta;
  NIXL_CHECK(agent_->getLocalMD(meta));
  return meta;
}

void NixlBackend::registerBuffer(void* buffer, size_t size) {
  nixlBlobDesc desc;
  desc.addr = (uintptr_t)buffer;
  desc.len = size;
  desc.devId = 0;
  reg_buffers_.addDesc(desc);

  NIXL_CHECK(agent_->registerMem(reg_buffers_, &extra_params_));
}

bool NixlBackend::send(void* buffer, size_t size, const std::string& tag, const std::string& peer_address) {
  Endpoint* endpoint = getPeerEndpoint(peer_address);

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(buffer);
  size_t size_10 = size / 10;

  // src desc.
  nixl_xfer_dlist_t req_src_descs(DRAM_SEG);
  for (size_t i = 0; i < 5; ++i) {
    nixlBasicDesc desc;
    desc.addr = base_addr + i * 2 * size_10;
    desc.len = size_10;
    desc.devId = 0;
    req_src_descs.addDesc(desc);
  }
  // nixlBasicDesc req_desc;
  // req_desc.addr = (uintptr_t)buffer;
  // req_desc.len = size / 3;
  // req_desc.devId = 0;
  // req_src_descs.addDesc(req_desc);
  // dst desc.
  nixlSerDes ser;
  while (!dist_store_.check(tag)) {
  }
  LogInfo("tag {} recv remote memory desc.", tag);

  ser.importStr(comm::HexToBinary(dist_store_.get(tag)));
  nixl_xfer_dlist_t req_dst_descs(&ser);

  extra_params_.notifMsg = tag;
  extra_params_.hasNotif = true;
  nixlXferReqH* req_handle;
  NIXL_CHECK(
      agent_->createXferReq(NIXL_WRITE, req_src_descs, req_dst_descs, endpoint->uuid, req_handle, &extra_params_));

  auto s = agent_->postXferReq(req_handle);
  while (s == NIXL_IN_PROG) {
    s = agent_->getXferStatus(req_handle);
  }

  NIXL_CHECK(agent_->releaseXferReq(req_handle));

  return s == NIXL_SUCCESS;
}

bool NixlBackend::recv(void* buffer, size_t size, const std::string& tag, const std::string& peer_address) {
  getPeerEndpoint(peer_address);

  uintptr_t base_addr = reinterpret_cast<uintptr_t>(buffer);
  size_t size_10 = size / 10;

  // dst desc.
  nixl_xfer_dlist_t recv_descs(DRAM_SEG);
  for (size_t i = 0; i < 5; ++i) {
    nixlBasicDesc desc;
    desc.addr = base_addr + i * 2 * size_10;
    desc.len = size_10;
    desc.devId = 0;
    recv_descs.addDesc(desc);
  }

  // nixl_xfer_dlist_t recv_descs(DRAM_SEG);
  // nixlBasicDesc desc;
  // desc.addr = (uintptr_t)buffer;
  // desc.len = size;
  // desc.devId = 0;
  // recv_descs.addDesc(desc);

  nixlSerDes ser;
  recv_descs.serialize(&ser);
  if (dist_store_.set(peer_address, tag, comm::BinaryToHex(ser.exportStr())) != comm::RetCode::OK) {
    return false;
  }
  LogInfo("tag {} send memory desc.", tag);

  nixl_notifs_t notif_map;
  while (notif_map.empty()) {
    NIXL_CHECK(agent_->getNotifs(notif_map));
  }

  return true;
}

NixlBackend::Endpoint* NixlBackend::getPeerEndpoint(const std::string& peer_address) {
  auto it = endpoints_.find(peer_address);
  if (it != endpoints_.end()) {
    return &it->second;
  }

  std::string local_metadata = getLocalMetadata();
  local_metadata = comm::BinaryToHex(local_metadata);
  while (dist_store_.set(peer_address, address_, local_metadata) != comm::RetCode::OK) {
    comm::TimestampApi::MilliSleep(1000);
  }
  LogInfo("set metadata to peer address {}", peer_address);
  while (!dist_store_.check(peer_address)) {
  }
  auto metadata = dist_store_.get(peer_address);
  metadata = comm::HexToBinary(metadata);
  LogInfo("get metadata of peer address, size {}", metadata.size());

  std::string peer_uuid;
  agent_->loadRemoteMD(metadata, peer_uuid);

  endpoints_.insert({peer_address, Endpoint{.uuid = peer_uuid}});

  return &endpoints_[peer_address];
}