#pragma once

#include <memory>

class Interface {
 public:
  virtual void fn1(int) = 0;
  virtual void fn2(float) = 0;
};

template <typename T>
class Wrapper : public Interface {
 public:
  Wrapper(T c) : interface_(c) {}

  void fn1(int x) override { return interface_.fn1(x); }
  void fn2(float x) override { return interface_.fn2(x); }

 private:
  T interface_;
};

class A {
 public:
  template <typename T>
  A(T c) : interface_(new Wrapper<T>(c)) {}

  void fn1(int x) { return interface_->fn1(x); }
  void fn2(float x) { return interface_->fn2(x); }

 private:
  std::unique_ptr<Interface> interface_;
};

inline void fn(A a) {
  a.fn1(0);
  a.fn2(0);
}