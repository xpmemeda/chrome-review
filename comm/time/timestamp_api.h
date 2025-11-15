#pragma once

#include <unistd.h>
#include <chrono>

namespace comm {

class TimestampApi {
 public:
  static const uint32_t NowSecond();
  static const uint64_t NowMilliSecond();
  static const uint64_t NowMicroSecond();

  static void MilliSleep(const int ms);
};

class IntervalTiming {
 public:
  IntervalTiming();

  int PassTimeFromStart();
  int PassTime();

  bool CheckHasPassAndReset(int check_pass_time);

 private:
  int64_t start_time_;
  int64_t mark_time_;
};

}  // namespace comm
