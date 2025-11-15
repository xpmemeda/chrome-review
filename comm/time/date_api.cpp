#include <sys/time.h>
#include <time.h>
#include <ctime>
#include <iomanip>

#include "date_api.h"

using namespace std::chrono;

namespace comm {

const std::string DateApi ::Now() {
  struct timeval now;
  gettimeofday(&now, NULL);
  auto seconds = now.tv_sec;
  struct tm tm_now;
  localtime_r(&seconds, &tm_now);
  char str_now[25] = {0};
  snprintf(str_now, sizeof(str_now), "%02d-%02d %02d:%02d:%02d.%03d", tm_now.tm_mon + 1, tm_now.tm_mday, tm_now.tm_hour,
      tm_now.tm_min, tm_now.tm_sec, (int)now.tv_usec / 1000);
  return std::string(str_now);
}

const std::string DateApi ::NowHour() {
  auto now = system_clock::now();
  auto t_now = system_clock::to_time_t(now);
  struct tm tm_now;
  localtime_r(&t_now, &tm_now);
  char str_now[16] = {0};
  snprintf(str_now, sizeof(str_now), "%04d%02d%02d%02d", tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday,
      tm_now.tm_hour);
  return std::string(str_now);
}

const int DateApi ::NowHourNum() {
  auto now = system_clock::now();
  auto t_now = system_clock::to_time_t(now);
  struct tm tm_now;
  localtime_r(&t_now, &tm_now);
  return tm_now.tm_hour;
}

const std::string DateApi ::NowDay() {
  auto now = system_clock::now();
  auto t_now = system_clock::to_time_t(now);
  struct tm tm_now;
  localtime_r(&t_now, &tm_now);
  char str_now[16] = {0};
  snprintf(str_now, sizeof(str_now), "%04d%02d%02d", tm_now.tm_year + 1900, tm_now.tm_mon + 1, tm_now.tm_mday);
  return std::string(str_now);
}

const bool DateApi ::IsInSameHour(const time_t time_a, const time_t time_b) { return time_a / 3600 == time_b / 3600; }

const std::string DateApi ::TimestampToDate(const int time_second) {
  struct tm tm_now;
  time_t s = (time_t)time_second;
  localtime_r(&s, &tm_now);
  char str_now[25] = {0};
  snprintf(str_now, sizeof(str_now), "%04d-%02d-%02d %02d:%02d:%02d", tm_now.tm_year + 1900, tm_now.tm_mon + 1,
      tm_now.tm_mday, tm_now.tm_hour, tm_now.tm_min, tm_now.tm_sec);
  return std::string(str_now);
}

const int DateApi ::GetDayByTimestamp(const int time_second) {
  struct tm tm_now;
  time_t s = (time_t)time_second;
  localtime_r(&s, &tm_now);
  return tm_now.tm_mday;
}

}  // namespace comm