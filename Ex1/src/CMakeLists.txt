find_package(CUDA REQUIRED)
cuda_add_executable(ex1 main.cu)

add_custom_target(
    run
    COMMAND ex1
    WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
)