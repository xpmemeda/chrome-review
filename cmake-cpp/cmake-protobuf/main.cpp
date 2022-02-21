#include <map>
#include <utility>
#include <iostream>
#include <addressbook.pb.h>

using namespace tutorial;

enum CppEnum {
    CppEnum_X = 1,
    CppEnum_Y = 2,
};

int main() {
    {
        CppEnum a;
        std::cout << "CppEnum a: " << a << std::endl;  // 0
        CppEnum b = static_cast<CppEnum>(3);
        std::cout << "CppEnum b: " << b << std::endl;  // 3
    }
    {
        PbEnum a;
        std::cout << "PbEnum a: " << a << std::endl;  // 0
        PbEnum b = static_cast<PbEnum>(3);
        std::cout << "PbEnum b: " << b << std::endl;  // 3
    }

    tutorial::AddressBook x;
    x.mutable_dimension()->set_value(100);
    std::cout << x.DebugString() << std::endl;
    return 0;
}