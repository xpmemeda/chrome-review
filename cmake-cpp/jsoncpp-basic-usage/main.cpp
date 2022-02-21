#include <iostream>
#include <sstream>

#include "json/json.h"

int main() {
  Json::Value addressbook;
  Json::Value person;
  person["name"] = "olafxiong";
  person["phone"] = 18621107363;
  addressbook["people"][0] = person;
  std::string json_string = addressbook.toStyledString();
  std::cout << "addressbook:\n" << json_string << std::endl;

  Json::Value addressbook_copy;
  std::istringstream(json_string) >> addressbook_copy;
  std::cout << "addressbook_copy:\n"
            << addressbook_copy.toStyledString() << std::endl;

  std::string name = addressbook_copy["people"][0]["name"].asString();
  uint64_t phone = addressbook_copy["people"][0]["phone"].asUInt64();
  std::cout << name << ": " << phone << std::endl;

  return 0;
}