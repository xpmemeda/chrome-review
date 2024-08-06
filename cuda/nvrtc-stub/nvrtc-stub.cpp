#include <cstddef>
#include <cstdint>
#include <stdexcept>

extern "C" {

typedef enum {
  NVRTC_SUCCESS = 0,
  NVRTC_ERROR_OUT_OF_MEMORY = 1,
  NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVRTC_ERROR_INVALID_INPUT = 3,
  NVRTC_ERROR_INVALID_PROGRAM = 4,
  NVRTC_ERROR_INVALID_OPTION = 5,
  NVRTC_ERROR_COMPILATION = 6,
  NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
  NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
  NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
  NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
  NVRTC_ERROR_INTERNAL_ERROR = 11,
  NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12
} nvrtcResult;

typedef struct _nvrtcProgram* nvrtcProgram;

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcVersion(int* major, int* minor) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog, const char* src, const char* name, int numHeaders,
    const char* const* headers, const char* const* includeNames) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
const char* nvrtcGetErrorString(nvrtcResult result) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return nullptr;
}

nvrtcResult __nvrtcCPEx(nvrtcProgram prog, int numOptions, const char* const* options, int numHeaders,
    const char* const* headers, const char* const* includeNames) {
  throw std::runtime_error("using nvrtc stub, please check wnr compiling env");
  return NVRTC_SUCCESS;
}
}
