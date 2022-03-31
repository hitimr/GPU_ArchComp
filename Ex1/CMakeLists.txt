cmake_minimum_required(VERSION 3.14)
project("GPU_Ex1")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")

add_subdirectory(src)