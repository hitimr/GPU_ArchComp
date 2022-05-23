find_package(CUDA REQUIRED)

cuda_add_executable(ex2 
    main.cu 
    kruskal.cu 
    sort.cu
    graph.cu
)



target_link_libraries(ex2
    ${Boost_LIBRARIES}
)

target_include_directories(ex2 PUBLIC
    "../include"
)

# Create Output Folder
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/out)

# make run
add_custom_target(
    run
    COMMAND ex2 --sort-kernel 0 --ouputfile_timings out/timing_results.csv
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/..
)

target_include_directories(ex2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lodepng")
