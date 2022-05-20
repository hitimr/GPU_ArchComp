find_package(CUDA REQUIRED)

cuda_add_executable(ex2 main.cu)

set(MINIMUM_BOOST_VERSION 1.63.0)

find_package(Boost ${MINIMUM_BOOST_VERSION} REQUIRED COMPONENTS program_options)
target_link_libraries(ex2
    ${Boost_LIBRARIES}
)


add_custom_target(
    run
    COMMAND ex2
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

target_include_directories(ex2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lodepng")
