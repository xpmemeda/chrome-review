#include <iostream>

#include <cpr/cpr.h>

int main() {
  // Details: https://docs.libcpr.org/introduction.html#get-requests
  cpr::Response g =
      cpr::Get(cpr::Url{"http://www.httpbin.org/get"},
               // Parameters will be formed ?key=value after url...
               cpr::Parameters{{"hello", "world"}, {"stay", "cool"}});
  std::cout << "url: " << g.url << std::endl;
  std::cout << "status_code: " << g.status_code << std::endl;
  std::cout << "header: " << g.header["content-type"] << std::endl;
  std::cout << "text:\n" << g.text << std::endl;

  std::cout << "=================" << std::endl;
  cpr::Response p = cpr::Post(cpr::Url{"http://www.httpbin.org/post"},
                              cpr::Payload{{"key", "value"}});
  std::cout << "url: " << p.url << std::endl;
  std::cout << "status_code: " << p.status_code << std::endl;
  std::cout << "header: " << p.header["content-type"] << std::endl;
  std::cout << "text:\n" << p.text << std::endl;

  std::cout << "==============================" << std::endl;
  cpr::Response p2 = cpr::Post(cpr::Url{"http://www.httpbin.org/post"},
                               cpr::Body{"This is raw POST data"},
                               // set header...
                               cpr::Header{{"Content-Type", "text/plain"}});
  std::cout << "text:\n" << p2.text << std::endl;

  return 0;
}