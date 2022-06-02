find_package(CUDA REQUIRED)

cuda_add_executable(ex2 
    main.cu 
    kruskal.cu 
    sort.cu
    graph.cu
    partition.cu
    edgelist.cu
    union_find.cu
)



target_link_libraries(ex2
    ${Boost_LIBRARIES}
)

target_include_directories(ex2 PUBLIC
    "../include"
)

# Create Output Folder
file(MAKE_DIRECTORY ${OUT_DIR})

# make run
add_custom_target(
    run
    COMMAND ex2 --sort-kernel 0 --ouputfile_timings ${OUT_DIR}/timing_results.csv
    WORKING_DIRECTORY ${ROOT_PROJ_DIR}
)

target_include_directories(ex2 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lodepng")
