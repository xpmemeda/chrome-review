#include <inttypes.h>
#include <stdio.h>
#include <unistd.h>

// https://www.man7.org/linux/man-pages/man3/sysconf.3.html
unsigned long long get_total_memory() {
  long long pages = sysconf(_SC_PHYS_PAGES);
  long long page_size = sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}

int main(int arc, char* argv[]) {
  printf("%.2fG\n", get_total_memory() / static_cast<double>(1024 * 1024 * 1024));
  return 0;
}
