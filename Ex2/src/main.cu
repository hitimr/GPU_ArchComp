#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <vector>

#include "partition_array.cu"

void print_vector(std::vector<int> &vec)
{
    for (auto i: vec)
        std::cout << i << ' ';

    std::cout << std::endl;
}

int main()
{
    // random number gen from https://stackoverflow.com/questions/21516575/fill-a-vector-with-random-numbers-c#:~:text=You%20can%20use%20std%3A%3A,specify%20the%20engine%20and%20distribution.
    // First create an instance of an engine.
    std::random_device rnd_device;


    // Specify the engine and distribution.
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_int_distribution<int> dist {0, 10};

    auto gen = [&dist, &mersenne_engine](){
                   return dist(mersenne_engine);
               };

    int size_vector = 20;
    std::vector<int> vec(size_vector);
    std::generate(begin(vec), end(vec), gen);

    std::vector<std::vector<int>> result(2);

    result = partition_on_condition(vec, 5);

    print_vector(vec);
    print_vector(result[0]);
    print_vector(result[1]);



    return 0; //comment branch
}