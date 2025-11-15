#pragma once

#include <chrono>
#include <string>

namespace comm {

class DateApi {
 public:
  static const std::string Now();
  static const std::string NowHour();
  static const int NowHourNum();
  static const std::string NowDay();
  static const bool IsInSameHour(const time_t time_a, const time_t time_b);
  static const std::string TimestampToDate(const int time_second);
  static const int GetDayByTimestamp(const int time_second);
};

}  // namespace comm