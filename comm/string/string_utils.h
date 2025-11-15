#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace comm {

std::string StringPrintf(const char* format, ...);

std::string BinaryToHex(const std::string& binary_data);
std::string HexToBinary(const std::string& hex_str);

uint64_t SimpleHashString(const std::string& str);

}  // namespace comm