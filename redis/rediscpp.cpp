#include "rediscpp.h"

#include <string.h>
#include <iostream>
#include <stdexcept>

#include <hiredis.h>

CppRedis::CppRedis(const std::string& redis_host, int redis_port) : CppRedis(redis_host, redis_port, "COMM") {}

CppRedis::CppRedis(const std::string& redis_host, int redis_port, const uuid_t& uuid) : uuid_(uuid), c_(nullptr) {
  c_ = redisConnect(redis_host.c_str(), redis_port);
  if (c_ == NULL || c_->err) {
    std::cerr << "init redis err." << std::endl;
    throw std::runtime_error("init redis err.");
  }
  printf("connect redis server [%s:%d] succ.\n", redis_host.c_str(), redis_port);
}

CppRedis::~CppRedis() { redisFree(c_); }

bool CppRedis::set(const std::string& k, const std::string& v, int ex, bool nx) {
  redisReply* reply;

  std::string uuid_k = uuid_ + k;

  if (nx) {
    reply = static_cast<redisReply*>(redisCommand(c_, "SET %s %s EX %d NX", uuid_k.c_str(), v.c_str(), ex));
  } else {
    reply = static_cast<redisReply*>(redisCommand(c_, "SET %s %s EX %d", uuid_k.c_str(), v.c_str(), ex));
  }

  if (reply->type == REDIS_REPLY_STATUS && strcmp(reply->str, "OK") == 0) {
    freeReplyObject(reply);
    return true;
  }

  freeReplyObject(reply);
  return false;
}

bool CppRedis::del(const std::string& k) {
  auto uuid_k = uuid_ + k;

  auto reply = static_cast<redisReply*>(redisCommand(c_, "DEL %s", uuid_k.c_str()));
  if (reply->type == REDIS_REPLY_INTEGER && reply->integer == 1) {
    freeReplyObject(reply);
    return true;
  }
  freeReplyObject(reply);
  return false;
}
std::string CppRedis::get(const std::string& k) {
  std::string uuid_k = uuid_ + k;

  auto reply = static_cast<redisReply*>(redisCommand(c_, "GET %s", uuid_k.c_str()));
  if (reply->type != REDIS_REPLY_STRING) {
    freeReplyObject(reply);
    return "";
  }

  std::string v = reply->str;
  freeReplyObject(reply);
  return v;
}

std::vector<std::string> CppRedis::mget(const std::vector<std::string>& ks) {
  std::vector<std::string> uuid_ks(ks.size());
  for (auto& k : ks) {
    uuid_ks.push_back(uuid_ + k);
  }

  std::vector<std::string> values;

  if (uuid_ks.empty()) {
    return values;
  }

  int argc = 1 + uuid_ks.size();
  std::vector<const char*> argv;
  std::vector<size_t> argvlen;

  argv.push_back("MGET");
  argvlen.push_back(4);

  for (const auto& key : uuid_ks) {
    argv.push_back(key.c_str());
    argvlen.push_back(key.length());
  }

  auto reply = static_cast<redisReply*>(redisCommandArgv(c_, argc, argv.data(), argvlen.data()));

  if (!reply) {
    return values;
  }

  if (reply->type == REDIS_REPLY_ARRAY) {
    for (size_t i = 0; i < reply->elements; ++i) {
      redisReply* element = reply->element[i];
      if (element->type == REDIS_REPLY_STRING) {
        values.emplace_back(element->str, element->len);
      } else if (element->type == REDIS_REPLY_NIL) {
        values.emplace_back("");
      } else {
        values.emplace_back("");
      }
    }
  }

  freeReplyObject(reply);

  return values;
}

bool CppRedis::msadd(const std::vector<std::string>& keys, const std::string& member) {
  std::vector<std::string> uuid_ks;
  for (auto& k : keys) {
    uuid_ks.push_back(uuid_ + k);
  }

  if (uuid_ks.empty()) {
    return true;
  }

  const char* script = R"(
        local member_to_add = ARGV[1]
        local added_counts = {}
        for i, key in ipairs(KEYS) do
            local count = redis.call('SADD', key, member_to_add)
            table.insert(added_counts, count)
        end
        -- 返回一个包含所有新增数量的数组
        return added_counts
    )";

  int argc = 3 + uuid_ks.size() + 1;
  std::vector<const char*> argv;
  std::vector<size_t> argvlen;

  argv.push_back("EVAL");
  argvlen.push_back(4);

  argv.push_back(script);
  argvlen.push_back(strlen(script));

  std::string num_keys_str = std::to_string(uuid_ks.size());
  argv.push_back(num_keys_str.c_str());
  argvlen.push_back(num_keys_str.length());

  for (const auto& key : uuid_ks) {
    argv.push_back(key.c_str());
    argvlen.push_back(key.length());
  }

  // ARGV[1] 是要添加的成员
  argv.push_back(member.c_str());
  argvlen.push_back(member.length());

  // 执行命令
  auto reply = static_cast<redisReply*>(redisCommandArgv(c_, argc, argv.data(), argvlen.data()));

  // 处理返回结果
  if (!reply || reply->type == REDIS_REPLY_ERROR) {
    if (reply) {
      std::cerr << "Redis Error in sadd: " << reply->str << std::endl;
      freeReplyObject(reply);
    } else {
      std::cerr << "Connection Error in sadd: " << c_->errstr << std::endl;
    }
    return false;
  }

  // 脚本成功执行，返回一个数组。
  // (可选) 打印添加详情用于调试
  if (reply->type == REDIS_REPLY_ARRAY) {
    std::cout << "sadd details: ";
    for (size_t i = 0; i < reply->elements; ++i) {
      std::cout << reply->element[i]->integer << " ";  // 打印每个集合新增的数量
    }
    std::cout << std::endl;
  }

  freeReplyObject(reply);
  return true;
}

bool CppRedis::msrem(const std::vector<std::string>& keys, const std::string& member) {
  std::vector<std::string> uuid_ks;
  for (auto& k : keys) {
    uuid_ks.push_back(uuid_ + k);
  }

  if (uuid_ks.empty()) {
    return true;  // 没有键需要操作，可以视为成功
  }

  // Lua 脚本
  const char* script = R"(
        local member_to_delete = ARGV[1]
        local deleted_counts = {}
        for i, key in ipairs(KEYS) do
            local count = redis.call('SREM', key, member_to_delete)
            table.insert(deleted_counts, count)
        end
        return deleted_counts
    )";

  // 准备 EVAL 命令的参数
  // EVAL <script> <num_keys> <key1> <key2> ... <arg1> ...
  int argc = 3 + uuid_ks.size() + 1;  // "EVAL", script, num_keys, keys..., member
  std::vector<const char*> argv;
  std::vector<size_t> argvlen;

  argv.push_back("EVAL");
  argvlen.push_back(4);

  argv.push_back(script);
  argvlen.push_back(strlen(script));

  std::string num_keys_str = std::to_string(uuid_ks.size());
  argv.push_back(num_keys_str.c_str());
  argvlen.push_back(num_keys_str.length());

  for (const auto& key : uuid_ks) {
    argv.push_back(key.c_str());
    argvlen.push_back(key.length());
  }

  // ARGV[1] 是要删除的成员
  argv.push_back(member.c_str());
  argvlen.push_back(member.length());

  // 执行命令
  auto reply = static_cast<redisReply*>(redisCommandArgv(c_, argc, argv.data(), argvlen.data()));

  // 处理返回结果
  if (!reply || reply->type == REDIS_REPLY_ERROR) {
    if (reply) {
      std::cerr << "Redis Error in sdel: " << reply->str << std::endl;
      freeReplyObject(reply);
    } else {
      std::cerr << "Connection Error in sdel: " << c_->errstr << std::endl;
    }
    return false;
  }

  // 脚本成功执行，返回一个数组。我们可以解析它来获取详细信息，但根据函数签名，我们只关心成功与否。
  // (可选) 打印删除详情用于调试
  if (reply->type == REDIS_REPLY_ARRAY) {
    std::cout << "sdel details: ";
    for (size_t i = 0; i < reply->elements; ++i) {
      std::cout << reply->element[i]->integer << " ";  // 打印每个集合删除的数量
    }
    std::cout << std::endl;
  }

  freeReplyObject(reply);
  return true;
}

std::vector<std::vector<std::string>> CppRedis::msget(const std::vector<std::string>& keys) {
  std::vector<std::vector<std::string>> all_results;

  std::vector<std::string> uuid_ks;
  for (auto& k : keys) {
    uuid_ks.push_back(uuid_ + k);
  }

  if (uuid_ks.empty()) {
    return all_results;
  }

  const char* script = R"(
        local results = {}
        for i, key in ipairs(KEYS) do
            local members = redis.call('SMEMBERS', key)
            table.insert(results, members)
        end
        return results
    )";

  int argc = 3 + uuid_ks.size();
  std::vector<const char*> argv;
  std::vector<size_t> argvlen;

  argv.push_back("EVAL");
  argvlen.push_back(4);

  argv.push_back(script);
  argvlen.push_back(strlen(script));

  std::string num_keys_str = std::to_string(uuid_ks.size());
  argv.push_back(num_keys_str.c_str());
  argvlen.push_back(num_keys_str.length());

  for (const auto& key : uuid_ks) {
    argv.push_back(key.c_str());
    argvlen.push_back(key.length());
  }

  auto reply = static_cast<redisReply*>(redisCommandArgv(c_, argc, argv.data(), argvlen.data()));

  if (!reply || reply->type != REDIS_REPLY_ARRAY) {
    if (reply) {
      freeReplyObject(reply);
    }
    return all_results;
  }

  for (size_t i = 0; i < reply->elements; ++i) {
    redisReply* set_reply = reply->element[i];
    std::vector<std::string> single_set_members;

    if (set_reply->type == REDIS_REPLY_ARRAY) {
      for (size_t j = 0; j < set_reply->elements; ++j) {
        redisReply* member_reply = set_reply->element[j];
        if (member_reply->type == REDIS_REPLY_STRING) {
          single_set_members.emplace_back(member_reply->str, member_reply->len);
        }
      }
    }
    all_results.push_back(single_set_members);
  }

  freeReplyObject(reply);
  return all_results;
}
