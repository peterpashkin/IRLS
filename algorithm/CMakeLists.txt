# Gather all source files in the current directory
file(GLOB ALGORITHM_SOURCES "*.cpp" "*.h")

# Create a library from the gathered sources
add_library(algorithm_library ${ALGORITHM_SOURCES})

# Optionally, include directories
target_include_directories(algorithm_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
