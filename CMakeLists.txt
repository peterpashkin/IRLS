cmake_minimum_required(VERSION 3.27)
project(l_infinity_testing)

# Find Eigen library
find_package(Eigen3 REQUIRED)

# Include Eigen
include_directories(${EIGEN3_INCLUDE_DIR})

# Set C++ standard and flags
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -INFINITY_FLAG -DLONG_STEP -DRECURSIVE_FLAG")

# Add subdirectories
add_subdirectory(algorithm)
add_subdirectory(graph_generation)

# Add the main executable
add_executable(l_infinity_testing testing.cpp)

# Link with libraries or targets from subdirectories if needed
target_link_libraries(l_infinity_testing algorithm_library graph_generation_library)
