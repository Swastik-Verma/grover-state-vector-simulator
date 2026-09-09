cmake_minimum_required(VERSION 3.16)
project(GroverSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Main executable
add_executable(grover src/main.cpp)
target_include_directories(grover PRIVATE include)
target_link_libraries(grover PRIVATE OpenMP::OpenMP_CXX)

# Test executables
add_executable(test_basic tests/test_basic.cpp)
target_include_directories(test_basic PRIVATE include)

add_executable(test_correctness tests/test_correctness.cpp)
target_include_directories(test_correctness PRIVATE include)
target_link_libraries(test_correctness PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_optimized_gates tests/test_optimized_gates.cpp)
target_include_directories(test_optimized_gates PRIVATE include)
target_link_libraries(test_optimized_gates PRIVATE OpenMP::OpenMP_CXX)

add_executable(test_day5_optimized tests/test_day5_optimized.cpp)
target_include_directories(test_day5_optimized PRIVATE include)
target_link_libraries(test_day5_optimized PRIVATE OpenMP::OpenMP_CXX)

# Benchmark
add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE include)
target_link_libraries(benchmark PRIVATE OpenMP::OpenMP_CXX)

# Day 7 OpenMP test
add_executable(test_day7_openmp tests/test_day7_openmp.cpp)
target_include_directories(test_day7_openmp PRIVATE include)
target_link_libraries(test_day7_openmp PRIVATE OpenMP::OpenMP_CXX)


# Day 8 cache test
add_executable(test_day8_cache tests/test_day8_cache.cpp)
target_include_directories(test_day8_cache PRIVATE include)
target_link_libraries(test_day8_cache PRIVATE OpenMP::OpenMP_CXX)


# Day 9 precision test
add_executable(test_day9_precision tests/test_day9_precision.cpp)
target_include_directories(test_day9_precision PRIVATE include)
target_link_libraries(test_day9_precision PRIVATE OpenMP::OpenMP_CXX)