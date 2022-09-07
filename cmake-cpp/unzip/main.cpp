#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stack>
#include <streambuf>
#include <utility>

#include <zlib.h>

static inline std::string _inflate(const char *data, size_t len,
                                   size_t targetLen) {
  size_t bufferSize = 1024 * 1024 * 1024;
  bufferSize = std::min(bufferSize, targetLen);
  std::unique_ptr<Bytef[]> buffer(new Bytef[bufferSize]);
  std::string result;
  result.reserve(targetLen);

  z_stream zs;
  memset(&zs, 0, sizeof(zs));
  int ret = inflateInit2(&zs, -8);
  if (ret != Z_OK) {
    inflateEnd(&zs);
    throw std::runtime_error("zip uncompress failed");
  }

  zs.next_in = const_cast<Bytef *>(reinterpret_cast<const Bytef *>(data));
  while (ret != Z_STREAM_END) {
    zs.avail_in = std::min(
        bufferSize, len - (zs.next_in - reinterpret_cast<const Bytef *>(data)));
    zs.avail_out = bufferSize;
    zs.next_out = buffer.get();
    ret = inflate(&zs, Z_NO_FLUSH);
    if (ret == Z_NEED_DICT || ret == Z_DATA_ERROR || ret == Z_MEM_ERROR) {
      inflateEnd(&zs);
      throw std::runtime_error("zip uncompress failed");
    }
    result.append(reinterpret_cast<char *>(buffer.get()),
                  bufferSize - zs.avail_out);
  }
  inflateEnd(&zs);

  return result;
}

static inline void _parse_local_file(
    const std::string &data, const std::size_t &local_file_header_pos,
    const std::size_t central_directory_pos, const uint16_t &compress_type,
    const std::string &name, const std::size_t &compressed_data_size,
    const std::size_t &uncompressed_data_size,
    std::map<std::string, std::string> &result) {
  if (local_file_header_pos + 30 - 1 >= central_directory_pos) {
    throw std::runtime_error(
        "invalid zip format, no enough space for local file header");
  }

  if (*reinterpret_cast<const uint32_t *>(
          data.data() + local_file_header_pos) != 0x04034b50) {
    throw std::runtime_error(
        "invalid zip format, Local File Header signature 0x04034b50 not found");
  }

  std::size_t name_size = *reinterpret_cast<const uint16_t *>(
      data.data() + local_file_header_pos + 26);
  std::size_t extra_data_size = *reinterpret_cast<const uint16_t *>(
      data.data() + local_file_header_pos + 28);
  if (local_file_header_pos + 30 + name_size + extra_data_size +
          compressed_data_size - 1 >=
      central_directory_pos) {
    throw std::runtime_error(
        "invalid zip format, no enough space to decompress data");
  }

  if (compress_type == 0) {
    result[name] = std::string(
        data.data() + local_file_header_pos + 30 + name_size + extra_data_size,
        data.data() + local_file_header_pos + 30 + name_size + extra_data_size +
            compressed_data_size);
  } else {
    result[name] = _inflate(data.data() + local_file_header_pos + 30 +
                                name_size + extra_data_size,
                            compressed_data_size, uncompressed_data_size);
  }
}

static inline void _parse_central_directory(
    const std::string &data, const std::size_t &central_directory_pos,
    const std::size_t &central_directory_cnt, const std::size_t &EOCD_pos,
    std::map<std::string, std::string> &result) {
  std::size_t search_pos = central_directory_pos;

  for (std::size_t i = 0; i < central_directory_cnt; ++i) {
    if (*reinterpret_cast<const uint32_t *>(data.data() + search_pos) !=
        0x02014b50) {
      throw std::runtime_error(
          "invalid zip format, Central Directory File Header "
          "signature 0x02014b50 not found");
    }

    if (search_pos + 46 - 1 >= EOCD_pos)
      throw std::runtime_error(
          "invalid zip format, no enough space for central "
          "directory file header");

    uint16_t compress_type =
        *reinterpret_cast<const uint16_t *>(data.data() + search_pos + 10);
    if (compress_type != 0 && compress_type != 8)
      throw std::runtime_error("invalid zip compress type: " +
                               std::to_string(compress_type));

    std::size_t compressed_data_size =
        *reinterpret_cast<const uint32_t *>(data.data() + search_pos + 20);
    std::size_t uncompressed_data_size =
        *reinterpret_cast<const uint32_t *>(data.data() + search_pos + 24);
    std::size_t name_size =
        *reinterpret_cast<const uint16_t *>(data.data() + search_pos + 28);
    std::size_t extra_data_size =
        *reinterpret_cast<const uint16_t *>(data.data() + search_pos + 30);
    std::size_t comment_size =
        *reinterpret_cast<const uint16_t *>(data.data() + search_pos + 32);

    std::size_t local_file_header_pos =
        *reinterpret_cast<const uint32_t *>(data.data() + search_pos + 42);

    // if uncompressed_data_size, compressed_data_size and local_file_header_pos
    // cann't be stored in 32 bit
    if (uncompressed_data_size == 0xffffffff ||
        compressed_data_size == 0xffffffff ||
        local_file_header_pos == 0xffffffff) {
      // need to read field which ==  0xffffffff in Zip64 extra field
      std::size_t extra_field_pos = search_pos + 46 + name_size;
      if (extra_field_pos + 2 + 2 - 1 >= EOCD_pos)
        throw std::runtime_error(
            "invalid zip format, no enough space for Zip64 extra field");

      if (*reinterpret_cast<const uint16_t *>(data.data() + extra_field_pos) !=
          0x0001) {
        throw std::runtime_error(
            "invalid zip format, signature of Zip64 extra field not found");
      }
      std::size_t extra_field_chunk_size = *reinterpret_cast<const uint16_t *>(
          data.data() + extra_field_pos + 2); // 8, 16, 24 or 28
      if (extra_field_pos + 2 + 2 + extra_field_chunk_size - 1 >= EOCD_pos) {
        throw std::runtime_error("invalid zip format, no enough space to read "
                                 "chunks in Zip64 extra field");
      }

      if (uncompressed_data_size == 0xffffffff) {
        if (extra_field_chunk_size < 8) {
          throw std::runtime_error(
              "invalid zip format, no enough space in Zip64 extra field"
              "to read uncompressed_data_size");
        }
        uncompressed_data_size = *reinterpret_cast<const uint64_t *>(
            data.data() + extra_field_pos + 4);
      }
      if (compressed_data_size == 0xffffffff) {
        if (extra_field_chunk_size < 16) {
          throw std::runtime_error(
              "invalid zip format, no enough space in Zip64 extra field"
              "to read compressed_data_size");
        }
        compressed_data_size = *reinterpret_cast<const uint64_t *>(
            data.data() + extra_field_pos + 12);
      }

      if (local_file_header_pos == 0xffffffff) {
        if (extra_field_chunk_size < 24) {
          throw std::runtime_error(
              "invalid zip format, no enough space in Zip64 extra field"
              "to read local_file_header_pos");
        }
        local_file_header_pos = *reinterpret_cast<const uint64_t *>(
            data.data() + extra_field_pos + 20);
      }
    }

    std::string name(data.data() + search_pos + 46,
                     data.data() + search_pos + 46 + name_size);

    _parse_local_file(data, local_file_header_pos, central_directory_pos,
                      compress_type, name, compressed_data_size,
                      uncompressed_data_size, result);
    search_pos += 46 + name_size + extra_data_size + comment_size;
  }
}

static inline std::size_t _locate_EOCD64(const std::string &data) {
  const std::string sign = "PK\x06\x06"; // 0x06064b50
  std::size_t sign_pos = std::string::npos;
  std::size_t search_pos = std::string::npos;

  while (1) {
    sign_pos = data.rfind(sign, search_pos);
    if (sign_pos == std::string::npos)
      break;

    if (sign_pos + 12 <=
        data.size()) // to read EOCD64_size, we need 12 bytes after sign_pos
    {
      std::size_t EOCD64_size =
          *reinterpret_cast<const uint64_t *>(data.data() + sign_pos + 4) + 12;

      // EOCD64 is also not necessarily the last record in the file.
      // A End of Central Directory Locator follows (an additional 20 bytes at
      // the end).
      if (sign_pos + EOCD64_size == data.size() ||
          sign_pos + EOCD64_size + 20 == data.size()) {
        break;
      }
    } else {
      // maybe zip format is wrong or there is "0x06064b50" in comment
      search_pos = sign_pos - 1;
    }
  }
  return sign_pos;
}

std::size_t _locate_EOCD32(const std::string &data) {
  const std::string sign = "PK\x05\x06"; // 0x06054b50
  std::size_t sign_pos = std::string::npos;
  std::size_t search_pos = std::string::npos;

  while (1) {
    sign_pos = data.rfind(sign, search_pos);
    if (sign_pos == std::string::npos)
      break;
    // check if it is a real EOCD or just there is "0x06054b50" in comment
    uint16_t comment_length =
        *reinterpret_cast<const uint16_t *>(data.data() + sign_pos + 20);
    if (sign_pos + 22 + static_cast<std::size_t>(comment_length) ==
        data.size()) {
      break;
    } else {
      // maybe zip format is wrong or there is "0x06054b50" in comment
      search_pos = sign_pos - 1;
    }
  }
  return sign_pos;
}

std::map<std::string, std::string> unzip(const std::string &data) {
  std::size_t EOCD_pos = _locate_EOCD32(data);
  std::size_t central_directory_pos = std::string::npos;
  std::size_t central_directory_cnt = 0;

  if (EOCD_pos != std::string::npos) {
    central_directory_cnt =
        *reinterpret_cast<const uint16_t *>(data.data() + EOCD_pos + 10);
    central_directory_pos =
        *reinterpret_cast<const uint32_t *>(data.data() + EOCD_pos + 16);
  }

  // if field value == 0xffffffff or 0xffff, the field value maybe store in
  // EOCD64
  if (central_directory_pos == 0xffffffff || central_directory_cnt == 0xffff ||
      EOCD_pos == std::string::npos) {
    EOCD_pos = _locate_EOCD64(data);
    if (EOCD_pos != std::string::npos) {
      if (central_directory_cnt == 0xffff)
        central_directory_cnt =
            *reinterpret_cast<const uint64_t *>(data.data() + EOCD_pos + 32);
      if (central_directory_pos == 0xffffffff)
        central_directory_pos =
            *reinterpret_cast<const uint64_t *>(data.data() + EOCD_pos + 48);
    } else {
      throw std::runtime_error("invalid zip format: cann't find end of central "
                               "directory EOCD32 or EOCD64");
    }
  }

  std::map<std::string, std::string> result;
  // read Central directory and parse local file
  _parse_central_directory(data, central_directory_pos, central_directory_cnt,
                           EOCD_pos, result);

  return result;
}

std::string load_data_from_file(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
    throw std::runtime_error("open file: \"" + path + "\" failed");
  ifs.seekg(0, ifs.end);
  auto length = ifs.tellg();
  ifs.seekg(0, ifs.beg);
  std::unique_ptr<char[]> buffer(new char[length]);
  ifs.read(buffer.get(), length);
  std::string data(buffer.get(), length);
  return data;
}

int main(int argc, char *argv[]) {
  auto x = unzip(load_data_from_file(argv[1]));
  for (const auto &i : x)
    std::cout << i.first << std::endl;
  return 0;
}