#include <addressbook.pb.h>
#include <fstream>
#include <iostream>
#include <map>
#include <utility>

using namespace tutorial;

/// Change the message name or field name will not break the compatibility.
void func_A() {
  A1 a1;
  a1.set_a1("hello-world-A");
  A2 a2;
  a2.ParseFromString(a1.SerializeAsString());
  std::cout << a2.a2() << std::endl;
}

// Reinterpreting a field as ``oneof`` does not break the compatibility.
void func_B() {
  B1 b1;
  b1.set_b1("hello-world-B");
  B2 b2;
  b2.ParseFromString(b1.SerializeAsString());
  std::cout << b2.b2_1() << std::endl;
}

// Reinterpreting a field as a new ``message`` will break the compatibility.
void func_C() {
  C1 c1;
  c1.set_c1("hello-world-C");
  C2 c2;
  c2.ParseFromString(c1.SerializeAsString());
  std::cout << c2.c2().c2_internal() << std::endl;
}

void func_D() { std::cout << static_cast<int>(EnumD()) << std::endl; }

// Reinterpreting a ``int32`` as ``int64`` or ``Enum`` does not break the compatibility.
void func_E() {
  E1 e1;
  e1.set_e1(101);
  E2 e2;
  e2.ParseFromString(e1.SerializeAsString());
  std::cout << e2.e2() << std::endl;
  E3 e3;
  e3.ParseFromString(e1.SerializeAsString());
  std::cout << static_cast<int>(e3.e3()) << std::endl;
}

// Reinterpreting a field to ``repeated`` does not break the compatibility.
void func_F() {
  F1 f1;
  f1.set_f1_1("hello-world-F");
  f1.set_f1_2(101);
  F2 f2;
  f2.ParseFromString(f1.SerializeAsString());
  std::cout << f2.f2_1_size() << std::endl;
  std::cout << f2.f2_1(0) << std::endl;
  std::cout << f2.f2_2_size() << std::endl;
  std::cout << f2.f2_2(0) << std::endl;
}

int main() {
  func_A();
  func_B();
  func_C();
  func_D();
  func_E();
  func_F();
  return 0;
}