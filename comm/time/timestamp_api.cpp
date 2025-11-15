#include <thread>

#include "timestamp_api.h"

using namespace std::chrono;

namespace comm {

const uint32_t TimestampApi ::NowSecond() {
  auto now_time = system_clock::now();
  return (duration_cast<seconds>(now_time.time_since_epoch())).count();
}

const uint64_t TimestampApi ::NowMilliSecond() {
  auto now_time = system_clock::now();
  return (duration_cast<milliseconds>(now_time.time_since_epoch())).count();
}

const uint64_t TimestampApi ::NowMicroSecond() {
  auto now_time = system_clock::now();
  return (duration_cast<microseconds>(now_time.time_since_epoch())).count();
}

void TimestampApi ::MilliSleep(const int ms) { std::this_thread::sleep_for(milliseconds(ms)); }

IntervalTiming ::IntervalTiming() {
  start_time_ = TimestampApi::NowMilliSecond();
  mark_time_ = start_time_;
}

int IntervalTiming ::PassTimeFromStart() {
  auto now_time = TimestampApi::NowMilliSecond();
  int pass_time = (int)(now_time - start_time_);
  return pass_time;
}

int IntervalTiming ::PassTime() {
  auto now_time = TimestampApi::NowMilliSecond();
  int pass_time = (int)(now_time - mark_time_);
  mark_time_ = now_time;
  return pass_time;
}

bool IntervalTiming ::CheckHasPassAndReset(int check_pass_time) {
  auto now_time = TimestampApi::NowMilliSecond();
  int pass_time = (int)(now_time - mark_time_);
  if (pass_time < check_pass_time) {
    return false;
  }
  mark_time_ = now_time;
  return true;
}

}  // namespace comm