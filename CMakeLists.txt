cmake_minimum_required(VERSION 3.16)
project(GroverSimulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Main executable
add_executable(grover src/main.cpp)
target_include_directories(grover PRIVATE include)

# Test executable
add_executable(test_basic tests/test_basic.cpp)
target_include_directories(test_basic PRIVATE include)


# Correctness test suite (Day 3)
add_executable(test_correctness tests/test_correctness.cpp)
target_include_directories(test_correctness PRIVATE include)