#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <string>

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

class Benchmarker
{
public:
  std::map<std::string, std::list<double>> timings;
  std::map<std::string, Timer> timers;

  Benchmarker(){};

  void start(std::string &&tag) { timers[tag] = Timer(); }

  void stop(std::string &&tag)
  {
    double time = timers[tag].get();
    timings[tag].push_back(time);
  }

  void print_timings()
  {
    std::map<std::string, std::list<double>>::iterator itr;
    for (itr = timings.begin(); itr != timings.end(); itr++)
    {
      double avg = average(itr->first);
      std::cout << itr->first << ":    " << avg <<  std::endl;
    }
  }

  double average(std::string tag)
  {
    auto data = timings[tag];
    double average = 0;

    for (double element : data)
    {
      average += element;
    }

    average /= (double)data.size();
    return average;
  }
};