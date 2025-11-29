#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>

#include <httplib.h>

struct MetadataStore {
  std::unordered_map<std::string, std::string> kv;
  std::mutex mu;

  void put(const std::string& key, const std::string& val) {
    std::lock_guard<std::mutex> g(mu);
    printf("Storing metadata: %s = %s\n", key.c_str(), val.c_str());
    kv[key] = val;
  }

  bool get(const std::string& key, std::string& out) {
    std::lock_guard<std::mutex> g(mu);
    auto it = kv.find(key);
    if (it == kv.end()) return false;
    out = it->second;
    printf("Retrieving metadata: %s = %s\n", key.c_str(), out.c_str());
    return true;
  }

  void erase(const std::string& key) {
    std::lock_guard<std::mutex> g(mu);
    printf("Erasing metadata: %s\n", key.c_str());
    kv.erase(key);
  }
};

int main(int argc, char** argv) {
  int port = 8080;
  if (argc > 1) {
    port = std::stoi(argv[1]);
  }

  MetadataStore store;
  httplib::Server svr;

  svr.Put("/metadata", [&store](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("key")) {
      res.status = 400;
      res.set_content("missing key", "text/plain");
      return;
    }
    auto key = req.get_param_value("key");
    auto& body = req.body;  // value 就是 request body

    store.put(key, body);
    res.status = 200;
    res.set_content("", "text/plain");
  });

  svr.Get("/metadata", [&store](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("key")) {
      res.status = 400;
      res.set_content("missing key", "text/plain");
      return;
    }
    auto key = req.get_param_value("key");
    std::string val;
    if (store.get(key, val)) {
      res.status = 200;
      res.set_content(val, "text/plain");
    } else {
      res.status = 404;
      printf("Metadata not found for key: %s\n", key.c_str());
      res.set_content("metadata not found", "text/plain");
    }
  });

  svr.Delete("/metadata", [&store](const httplib::Request& req, httplib::Response& res) {
    if (!req.has_param("key")) {
      res.status = 400;
      res.set_content("missing key", "text/plain");
      return;
    }
    auto key = req.get_param_value("key");
    store.erase(key);
    res.status = 200;
    res.set_content("", "text/plain");
  });

  std::cout << "HTTP metadata server (cpp-httplib) listening on 0.0.0.0:" << port << "\n";

  svr.listen("0.0.0.0", port);

  return 0;
}
