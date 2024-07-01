# Gather all source files in the current directory
file(GLOB GRAPH_SOURCES "*.cpp" "*.h")

# Create a library from the gathered sources
add_library(graph_generation_library ${GRAPH_SOURCES})

# Optionally, include directories
target_include_directories(graph_generation_library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
