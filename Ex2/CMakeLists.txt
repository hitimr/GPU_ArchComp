cmake_minimum_required(VERSION 3.14)
project("GPU_Ex2")


set(MINIMUM_BOOST_VERSION 1.63.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall")

set(ROOT_PROJ_DIR ${CMAKE_SOURCE_DIR})
set(OUT_DIR ${ROOT_PROJ_DIR}/out)

find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS program_options)
find_package(CUDA REQUIRED)


message("Run directory: ${CMAKE_SOURCE_DIR}")

add_subdirectory(src)




# Tests
option(BUILD_TESTS "Build unit tests." ON)
if(BUILD_TESTS)
  # Activate CTest testing framework
  enable_testing()
  # Add all tests in folder "tests" to the testing
  add_subdirectory("test")
endif()
