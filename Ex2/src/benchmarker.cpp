#include <benchmarker.hpp>

#include <cmath>
#include <iostream>

Timer::Timer() { reset(); }

double Timer::get()
{
  auto end_time = std::chrono::high_resolution_clock::now();
  double duration =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - chrono_ts).count() * 1e-9;

  return duration;
}

void Timer::reset() { chrono_ts = std::chrono::high_resolution_clock::now(); }

void Benchmarker::print_timings()
{
  std::map<std::string, std::list<double>>::iterator itr;
  for (itr = timings.begin(); itr != timings.end(); itr++)
  {
    auto tag = itr->first;
    std::cout << tag << "s\tµ=" << average(tag) << "s\tsigma=" << std_deviation(tag)
              << "s\ttotal=" << sum(tag) << std::endl;
  }
}

void Benchmarker::start(std::string &&tag) { timers[tag] = Timer(); }

void Benchmarker::stop(std::string &&tag)
{
  double time = timers[tag].get();
  timings[tag].push_back(time);
}

double Benchmarker::average(std::string tag) { return sum(tag) / (double)timings[tag].size(); }

double Benchmarker::std_deviation(std::string tag)
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

double Benchmarker::sum(std::string tag)
{
  double sum = 0;
  for (double element : timings[tag])
  {
    sum += element;
  }
  return sum;
}