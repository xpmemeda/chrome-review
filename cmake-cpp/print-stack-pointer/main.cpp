#include <iostream>

void print_stack_pointer(int x) {
  void *p = NULL;
  printf("%i, %p\n", x, (void *)&p);
}

int func(int x) {
  print_stack_pointer(x);
  if (x >= 10)
    return 0;
  char arr[1u << 20];
  // arr may be optimized if not use.
  std::cout << (int)arr[0] << std::endl;
  return func(x + 1);
}

int main() {
  std::cout << func(0) << std::endl;
  return 0;
}