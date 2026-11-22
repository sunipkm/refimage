# A C++ program against an installed refimage, through its CMake package
# (<prefix>/lib/cmake/refimage, from crates/refimage-capi/install.sh):
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<prefix>
#   cmake --build build
#   build/stream && build/stream_static
cmake_minimum_required(VERSION 3.16)
project(refimage_cpp_example CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(refimage 1.0 REQUIRED)

# Against the shared library ...
add_executable(stream stream.cpp)
target_link_libraries(stream PRIVATE refimage::refimage)

# ... and the static one (the target carries its system libraries).
add_executable(stream_static stream.cpp)
target_link_libraries(stream_static PRIVATE refimage::refimage_static)
