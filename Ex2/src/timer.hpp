#pragma once
#include <chrono>
#include <vector>

class Timer
{
public:
  Timer() { reset(); }

  void reset() { chrono_ts = std::chrono::high_resolution_clock::now(); }

  double get()
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - chrono_ts).count() * 1e-9;

    return duration;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> chrono_ts;
};

