#pragma once

#include <string>
#include <vector>

class redisContext;

using uuid_t = std::string;

class CppRedis {
 public:
  CppRedis(const std::string& redis_host, int redis_port);
  CppRedis(const std::string& redis_host, int redis_port, const uuid_t& uuid);
  ~CppRedis();

  bool set(const std::string& k, const std::string& v, int ex, bool nx);
  bool del(const std::string& k);
  std::string get(const std::string& k);
  std::vector<std::string> mget(const std::vector<std::string>& ks);

  bool msadd(const std::vector<std::string>& keys, const std::string& member);
  bool msrem(const std::vector<std::string>& keys, const std::string& member);
  std::vector<std::vector<std::string>> msget(const std::vector<std::string>& keys);

 private:
  uuid_t uuid_;
  redisContext* c_;
};
