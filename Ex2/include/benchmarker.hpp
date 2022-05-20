#pragma once
#include <chrono>
#include <list>
#include <map>
#include <string>


/*
  Very basic Timer-Class to keep track of time
*/
class Timer
{
public:
  Timer();

  // set timer to 0
  void reset();

  // get the time in seconds since the last reset
  double get();

private:
  std::chrono::time_point<std::chrono::high_resolution_clock> chrono_ts;
};


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
class Benchmarker
{
public:
  Benchmarker(){};

  // start timer for a given tag
  void start(std::string &&tag);

  // stop timer for a given tag and store the elapsed time
  void stop(std::string &&tag);

  // Print all Timing results to console
  void print_timings();

  // return the average time of a given tag
  double average(std::string tag);

  // return the standard deviation of a given tag
  double std_deviation(std::string tag);

  // return the total time of a given tag
  double sum(std::string tag);

private:
  std::map<std::string, std::list<double>> timings;
  std::map<std::string, Timer> timers;
};