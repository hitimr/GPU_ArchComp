cmake_minimum_required(VERSION 3.0)


add_custom_target(myTests)


set(MY_TEST_NAMES "")
list(APPEND MY_TEST_NAMES "test_sort")
list(APPEND MY_TEST_NAMES "test_partition")
list(APPEND MY_TEST_NAMES "test_edgeList")


foreach(MY_TEST_NAME ${MY_TEST_NAMES})
    cuda_add_executable(${MY_TEST_NAME} ${MY_TEST_NAME}.cu 
        ../src/edgelist.cu 
        ../src/sort.cu
        ../src/partition.cu
    )
    # target_link_libraries(${MY_TEST_NAME} PRIVATE ${MY_LIBRARY})

    target_include_directories(${MY_TEST_NAME} PUBLIC
    "../include"
    )

    target_link_libraries(${MY_TEST_NAME}
        ${Boost_LIBRARIES}
    )

    # Add to test framework, so test can be built with `make myTests`
    # add_dependencies(myTests ${MY_TEST_NAME})
    add_test(NAME ${MY_TEST_NAME} COMMAND ${MY_TEST_NAME})
    add_test(NAME ${MY_TEST_NAME}_pinned COMMAND ${MY_TEST_NAME} --pinned-memory)
endforeach(MY_TEST_NAME ${MY_TEST_NAMES})