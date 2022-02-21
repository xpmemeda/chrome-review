#include <map>
#include <utility>
#include <iostream>
#include <addressbook.pb.h>

int main() {
    tutorial::AddressBook x;
    x.mutable_dimension()->set_value(100);
    std::cout << x.DebugString() << std::endl;
    return 0;
}