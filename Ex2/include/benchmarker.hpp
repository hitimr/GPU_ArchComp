#pragma once
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <stdexcept>
#include <string>

/*
  Benchmarker for tracking multiple timers

  Example usage:

    Benchmarker b;

    using namespace std::chrono_literals;
    b.start("sleep1");
    b.start("sleep2");
    std::this_thread::sleep_for(100ms);
    b.stop("sleep1");


    b.start("sleep1");
    std::this_thread::sleep_for(100ms);
    b.stop("sleep1");
    b.stop("sleep2");

    b.print_timings();

  possible output:

    >>>> sleep1s µ=0.100062s     sigma=9.21007e-07s   total=0.200123
    >>>> sleep2s µ=0.200125s     sigma=0s             total=0.200125

*/

/*
  Very basic Timer-Class to keep track of time
*/
class Timer
{
public:
  Timer() { reset(); }

  // get the time in seconds since the last reset
  double get()
  {
    auto end_time = std::chrono::high_resolution_clock::now();
    double duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - chrono_ts).count() * 1e-9;

    return duration;
  }

  // set timer to 0
  void reset() { chrono_ts = std::chrono::high_resolution_clock::now(); }

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> chrono_ts;
};

class Benchmarker
{
public:
  Benchmarker(){};

  // start timer for a given tag
  void start(std::string &&tag) { timers[tag] = Timer(); }

  // stop timer for a given tag and store the elapsed time

  void stop(std::string &&tag)
  {
    double time = timers[tag].get();
    timings[tag].push_back(time);
  }

  // Print all Timing results to console
  void print_timings()
  {
    std::map<std::string, std::list<double>>::iterator itr;
    for (itr = timings.begin(); itr != timings.end(); itr++)
    {
      auto tag = itr->first;
      std::cout << tag << "\tµ=" << average(tag) << "s\tsigma=" << std_deviation(tag)
                << "s\tmedian=" << median(tag) << "s\ttotal=" << sum(tag)
                << "\tnum_calls=" << num_calls(tag) << std::endl;
    }
  }

  void export_csv(std::string filename)
  {
    std::ofstream file;
    file.open(filename);
    if (!file.is_open())
    {
      throw std::runtime_error("Unable to open output " + filename);
    }

    // Header
    file << "tag;average;std_dev;median;total;num_calls" << std::endl;

    // Data
    for (auto itr = timings.begin(); itr != timings.end(); itr++)
    {
      // clang-format off
      auto tag = itr->first;
      file  << tag << ";" 
            << average(tag) << ";" 
            << std_deviation(tag) << ";"
            << median(tag) << ";"
            << sum(tag) << ";"
            << num_calls(tag) << std::endl;
      // clang-format on
    }

    file.close();
  }

  size_t num_calls(std::string tag) { return timings[tag].size(); }

  // return the average time of a given tag
  double average(std::string tag) { return sum(tag) / (double)timings[tag].size(); }

  double median(std::string tag)
  {
    auto data = timings[tag];
    auto itr = data.begin();
    std::advance(itr, data.size() / 2);
    return *itr;
  }

  // return the standard deviation of a given tag
  double std_deviation(std::string tag)
  {
    auto data = timings[tag];
    double avg = average(tag);
    double sigma2 = 0;

    for (double element : data)
    {
      sigma2 += (element - avg) * (element - avg);
    }

    double sigma = sqrt(sigma2) / (double)data.size();
    return sigma;
  }

  // return the total time of a given tag
  double sum(std::string tag)
  {
    double sum = 0;
    for (double element : timings[tag])
    {
      sum += element;
    }
    return sum;
  }

private:
  std::map<std::string, std::list<double>> timings;
  std::map<std::string, Timer> timers;
};