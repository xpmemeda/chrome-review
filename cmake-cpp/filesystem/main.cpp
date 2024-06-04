#include <iostream>

#ifdef __GNUC__
#  include <features.h>
// the experimental is no longer needed in GCC 8+.
#  if __GNUC_PREREQ(8, 0)
#    include <filesystem>
namespace fs = std::filesystem;
#  else
#    include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#  endif
#else
#endif

enum class PathType { Directory, File, Unknown };

PathType determine_path_type(const std::string& path) {
  fs::path fsPath(path);

  if (fs::exists(fsPath)) {
    if (fs::is_regular_file(fsPath)) {
      return PathType::File;
    } else if (fs::is_directory(fsPath)) {
      return PathType::Directory;
    }
  }

  return PathType::Unknown;
}

std::vector<std::string> get_all_files_in_dir(const std::string& dirPath) {
  std::vector<std::string> filePaths;
  for (const auto& entry : fs::directory_iterator(dirPath)) {
    filePaths.push_back(entry.path().string());
  }
  return filePaths;
}

int main() {
  const char* path = "../main.cpp";

  if (fs::exists(path)) {
    std::cout << fs::is_directory(path) << std::endl;
    std::cout << fs::is_regular_file(path) << std::endl;
    std::cout << fs::file_size(path) << std::endl;
  } else
    std::cout << "not found main.cpp" << std::endl;
  std::cout << fs::current_path() << std::endl;

  for (const auto& dirEntry : fs::recursive_directory_iterator(".")) {
    std::cout << dirEntry.path().c_str() << std::endl;
  }

  for (const auto& dirEntry : fs::directory_iterator("..")) std::cout << dirEntry << std::endl;
  return 0;
}
