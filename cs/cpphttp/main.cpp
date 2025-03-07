#include <iostream>
#include <nlohmann/json.hpp>
#include "httplib.h"

using json = nlohmann::json;

void test_cli() {
  httplib::Client cli("127.0.0.1", 8000);

  std::string body = R"({"data": "x"})";

  httplib::Headers headers = {{"Content-Type", "application/json"}};

  bool result =
      cli.Post("/v1/chat/completions", headers, body, "application/json", [](const char* data, size_t data_length) {
        std::cout << "Received chunk: ";
        std::cout.write(data, data_length);
        std::cout << std::endl;
        return true;
      });

  if (!result) {
    std::cerr << "Request failed!" << std::endl;
  } else {
    std::cout << "Request finished." << std::endl;
  }
}

void test_svr() {
  httplib::Server svr;

  svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
    res.set_content_provider("text/plain", [](size_t offset, httplib::DataSink& sink) {
      for (int i = 0; i < 5; ++i) {
        std::string chunk = "chunk " + std::to_string(i) + "\n";
        sink.write(chunk.data(), chunk.size());
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }
      sink.done();
      return true;
    });
  });

  svr.Post("/wecube", [](const httplib::Request& req, httplib::Response& res) {
    std::cout << "Received data: " << req.body << std::endl;

    auto content = json::parse(req.body);
    if (content.contains("age")) {
      std::cout << "age: " << content["age"].template get<int>() << std::endl;
    }
    if (content.contains("name")) {
      std::cout << "name: " << content["name"].template get<std::string>() << std::endl;
    }
    std::cout << "Received data: " << content.dump() << std::endl;
  });

  std::cout << "Server listening on http://localhost:8080/v1/chat/completions" << std::endl;
  svr.listen("0.0.0.0", 8080);
}

void test_proxy() {
  httplib::Server svr;

  svr.Post("/v1/chat/completions", [](const httplib::Request& req, httplib::Response& res) {
    res.set_content_provider("application/json", [&](size_t offset, httplib::DataSink& sink) {
      httplib::Client cli("127.0.0.1", 8000);

      auto r = cli.Post("/v1/chat/completions", req.headers, req.body, "application/json",
          [&sink](const char* data, size_t data_length) {
            sink.write(data, data_length);
            return true;
          });

      if (!r || r->status != 200) {
        return false;
      }

      sink.done();

      return true;
    });
  });

  std::cout << "Server listening on http://localhost:8080/v1/chat/completions" << std::endl;
  svr.listen("0.0.0.0", 8080);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [cli|svr|proxy]" << std::endl;
    return 1;
  }

  if (argv[1] == std::string("cli")) {
    test_cli();
    return 0;
  } else if (argv[1] == std::string("svr")) {
    test_svr();
    return 0;
  } else if (argv[1] == std::string("proxy")) {
    test_proxy();
    return 0;
  } else {
    std::cerr << "Unknown mode: " << argv[1] << std::endl;
    return 1;
  }
}