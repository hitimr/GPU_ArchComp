find_package(CUDA REQUIRED)

cuda_add_executable(ex1 main.cu input.hpp lodepng/lodepng.cpp)


add_library(
    lodepng SHARED
    lodepng/lodepng.cpp
    lodepng/lodepng.h
)

add_custom_target(
    run
    COMMAND ex1
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_include_directories(ex1 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lodepng")
