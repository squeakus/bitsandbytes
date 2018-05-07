cmake_minimum_required(VERSION 3.5)
 
# Locate GTest
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
 
# Link runTests with what we want to test and the GTest and pthread library
add_executable(sample0_test tests/sample0_test.cpp)
target_link_libraries(sample0_test ${GTEST_LIBRARIES} pthread)
add_executable(sample1_test tests/sample1_test.cpp)
target_link_libraries(sample1_test ${GTEST_LIBRARIES} pthread)
add_executable(sample2_test tests/sample2_test.cpp)
add_library(sample2 src/sample2.cpp)
target_link_libraries(sample2_test ${GTEST_LIBRARIES} pthread sample2)
