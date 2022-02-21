#include <iostream>

/*
template <typename T, bool = true>
void func();
*/
template <typename T, typename std::enable_if<std::is_floating_point<T>::value,
                                              bool>::type = true>
void func() {
  std::cout << 0 << std::endl;
}

/*
template <typename T, typename U = void>
void func();

note:
  1) ``typename = `` equals to `` typename U = ``

  2) ``template <typename T, typename U = void>``
     and
     ``template <typename T, typename U = int>``
     will cause redefination.

    but
    ``template <typename T, bool = false>``
    and
    ``template <typename T, int = 0>``
    will not cause redefination.
*/
template <typename T, typename = std::enable_if_t<std::is_integral_v<T> &&
                                                  !std::is_unsigned_v<T>>>
void func() {
  std::cout << 1 << std::endl;
}

/*
template <typename T, int = 0>
void func();
*/
template <typename T,
          typename std::enable_if<std::is_integral_v<T> && !std::is_signed_v<T>,
                                  int>::type = 0>
void func() {
  std::cout << 2 << std::endl;
}

int main() {
  func<float>();        // 0
  func<int>();          // 1
  func<unsigned int>(); // 2
  return 0;
}
