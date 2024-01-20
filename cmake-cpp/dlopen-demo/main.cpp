#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "examplelib.h"

std::string get_lib_realpath(void* handle) {
  link_map* p;
  dlinfo(handle, RTLD_DI_LINKMAP, &p);
  std::string real_path(256, '\0');
  realpath(p->l_name, real_path.data());
  return real_path.substr(0, strlen(real_path.c_str()));
}

std::string exec(const char* cmd) {
  std::shared_ptr<FILE> pipe(popen(cmd, "r"), pclose);
  if (!pipe) return "ERROR";
  char buffer[128];
  std::string result = "";
  while (!feof(pipe.get())) {
    if (fgets(buffer, 128, pipe.get()) != NULL) result += buffer;
  }
  return result;
}

std::string get_symbol_mangle_name(const std::string& libpath, const std::string& raw_name) {
  std::string cmd;
  cmd.resize(1024);
  sprintf(cmd.data(), "nm -D %s| grep $(nm -CD %s | grep %s | awk '{print $1}') | awk '{print $3}'", libpath.c_str(),
      libpath.c_str(), raw_name.c_str());
  auto r = exec(cmd.c_str());
  return r.substr(0, r.size() - 1);
}

int main() {
  auto handle = dlopen("libexample.so", RTLD_LAZY);
  if (!handle) {
    throw std::runtime_error("cannot open lib");
  }
  std::string libpath = get_lib_realpath(handle);
  std::cout << "library real path: " << libpath << std::endl;

  auto example_func_ptr =
      reinterpret_cast<decltype(example_func)*>(dlsym(handle, get_symbol_mangle_name(libpath, "example_func").c_str()));
  if (!example_func_ptr) {
    throw std::runtime_error("cannot find function");
  }

  Params params({0, 0});
  Returns r = example_func_ptr(params);

  std::cout << r.x << ' ' << r.y << std::endl;

  return 0;
}