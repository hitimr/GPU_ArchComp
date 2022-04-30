find_package(CUDA REQUIRED)

cuda_add_executable(ex2 main.cu)


add_custom_target(
    run
    COMMAND ex2
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_include_directories(ex2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lodepng")
