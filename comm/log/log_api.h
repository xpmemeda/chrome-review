#pragma once

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

namespace comm {

#define LogInfo(...) SPDLOG_INFO(__VA_ARGS__)
#define LogError(...) SPDLOG_ERROR(__VA_ARGS__)
#define LogDebug(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LogHead(...) SPDLOG_INFO(__VA_ARGS__)

}  // namespace comm
