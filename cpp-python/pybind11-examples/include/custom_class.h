#ifndef _CUSTOM_CLASS_H_
#define _CUSTOM_CLASS_H_


#include <string>
#include <pybind11/pybind11.h>


struct Pet {
    Pet(const std::string& name);
    void setName(const std::string& name);
    const std::string& getName() const;
    std::string public_name;
  private:
    std::string _name;
};

struct Dog : Pet {
    Dog(const std::string& name);
    std::string bark();
};


void init_custom_class(pybind11::module m);

#endif
