#define _GNU_SOURCE

#include <dlfcn.h>
#include <stdio.h>

static void* (*real_malloc)(size_t) = NULL;

static void mtrace_init(void) {
  real_malloc = reinterpret_cast<decltype(real_malloc)>(dlsym(RTLD_NEXT, "malloc"));
  if (NULL == real_malloc) {
    fprintf(stderr, "Error in `dlsym`: %s\n", dlerror());
  }
}

extern "C" {

void* malloc(size_t size) {
  if (real_malloc == NULL) {
    mtrace_init();
  }

  void* p = NULL;
  p = real_malloc(size);
  printf("malloc(%d) = %p\n", size, p);
  return p;
}
}
