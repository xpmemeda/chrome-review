#include <chrono>
#include <thread>

#include <arpa/inet.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <sys/socket.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <unistd.h>
#include <boost/program_options.hpp>
#include <iostream>

namespace po = boost::program_options;

#define NCCLCHECK(cmd)                                                              \
  do {                                                                              \
    ncclResult_t r = cmd;                                                           \
    if (r != ncclSuccess) {                                                         \
      printf("NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                           \
    }                                                                               \
  } while (0)

#define CUDARTCHECK(cmd)                                                              \
  do {                                                                                \
    cudaError_t r = cmd;                                                              \
    if (r != cudaSuccess) {                                                           \
      printf("CUDART error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(r)); \
      exit(EXIT_FAILURE);                                                             \
    }                                                                                 \
  } while (0)

const int MAX_NODES = 8;

void sendUniqueId(int sock, ncclUniqueId* id) { send(sock, id->internal, NCCL_UNIQUE_ID_BYTES, 0); }

void recvUniqueId(int sock, ncclUniqueId* id) { recv(sock, id->internal, NCCL_UNIQUE_ID_BYTES, MSG_WAITALL); }

ncclUniqueId ncclSwapID(const std::string& master_ip, int port, int rank, int world_size) {
  ncclUniqueId id;

  if (rank == 0) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    int opt = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    ncclGetUniqueId(&id);
    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
      perror("bind error.");
      close(sock);
      exit(1);
    }
    if (listen(sock, MAX_NODES) != 0) {
      perror("listen error.");
      close(sock);
      exit(1);
    }

    for (int i = 1; i < world_size; i++) {
      int client = accept(sock, NULL, NULL);
      sendUniqueId(client, &id);
      close(client);
    }
    close(sock);
  } else {
    while (true) {
      int sock = socket(AF_INET, SOCK_STREAM, 0);
      struct sockaddr_in addr;
      addr.sin_family = AF_INET;
      addr.sin_port = htons(port);
      inet_pton(AF_INET, master_ip.c_str(), &addr.sin_addr);
      if (connect(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
        recvUniqueId(sock, &id);
        break;
      }
      if (errno == ECONNREFUSED || errno == ETIMEDOUT) {
        close(sock);
        std::cerr << "connecting..." << std::endl;
        sleep(1);
      } else {
        std::cerr << "connect error." << std::endl;
        exit(1);
      }
    }
  }

  return id;
}

po::variables_map parse_cmd_arguments(int argc, char* argv[]) {
  po::options_description desc("");
  auto add_options = desc.add_options();
  add_options("help", "produce help message");
  add_options("device", po::value<int>()->required(), "GPU index to use.");
  add_options("world-size", po::value<int>()->required(), "world size.");
  add_options("rank", po::value<int>()->required(), "rank.");
  add_options("master", po::value<std::string>()->required(), "master ip.");
  add_options("port", po::value<int>()->default_value(2009), "port.");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << std::endl;
    exit(0);
  }
  po::notify(vm);
  return vm;
}

void allReduceWorker(int device, const std::string& master_ip, int port, int rank, int world_size) {
  CUDARTCHECK(cudaSetDevice(device));

  ncclUniqueId id = ncclSwapID(master_ip, port, rank, world_size);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));

  cudaStream_t stream = 0;

  size_t count = 1 << 30;
  thrust::device_vector<float> a(count);
  thrust::fill(a.begin(), a.end(), 1.f);

  if (rank == 1) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  printf("ncclAllReduce ...\n");

  NCCLCHECK(ncclAllReduce(a.data().get(), a.data().get(), count, ncclFloat, ncclSum, comm, stream));
  CUDARTCHECK(cudaStreamSynchronize(stream));

  float max_v = *thrust::max_element(a.begin(), a.end());
  float min_v = *thrust::min_element(a.begin(), a.end());

  printf("allReduceWorker: rank=%i, max_v=%f, min_v=%f\n", rank, max_v, min_v);

  ncclCommDestroy(comm);
}

void sendRecvWorker(int device, const std::string& master_ip, int port, int rank, int world_size) {
  CUDARTCHECK(cudaSetDevice(device));

  ncclUniqueId id = ncclSwapID(master_ip, port + 1, rank, world_size);

  ncclComm_t comm;
  NCCLCHECK(ncclCommInitRank(&comm, world_size, id, rank));

  cudaStream_t stream;
  CUDARTCHECK(cudaStreamCreate(&stream));

  size_t count = 1 << 30;
  thrust::device_vector<float> a(count);

  if (rank == 0) {
    thrust::fill(a.begin(), a.end(), 1.f);
  }
  if (rank == 0) {
    std::this_thread::sleep_for(std::chrono::seconds(10));
  }

  printf("ncclSend/Recv ...\n");

  if (rank == 0) {
    NCCLCHECK(ncclSend(a.data().get(), count, ncclFloat, 1, comm, stream));
  } else {
    NCCLCHECK(ncclRecv(a.data().get(), count, ncclFloat, 0, comm, stream));
  }
  CUDARTCHECK(cudaStreamSynchronize(stream));

  float max_v = *thrust::max_element(a.begin(), a.end());
  float min_v = *thrust::min_element(a.begin(), a.end());

  printf("sendRecvWorker: rank=%i, max_v=%f, min_v=%f\n", rank, max_v, min_v);

  ncclCommDestroy(comm);
}

int main(int argc, char* argv[]) {
  auto cmd_arguments = parse_cmd_arguments(argc, argv);
  auto device = cmd_arguments["device"].as<int>();
  auto rank = cmd_arguments["rank"].as<int>();
  auto world_size = cmd_arguments["world-size"].as<int>();
  auto master_ip = cmd_arguments["master"].as<std::string>();
  auto port = cmd_arguments["port"].as<int>();

  auto t0 = std::thread(allReduceWorker, device, master_ip, port, rank, world_size);
  auto t1 = std::thread(sendRecvWorker, device, master_ip, port, rank, world_size);

  t0.join();
  t1.join();

  return 0;
}
