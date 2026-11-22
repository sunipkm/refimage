# Build the example against an installed refimage:
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<prefix> && cmake --build build
cmake_minimum_required(VERSION 3.16)
project(refimage_c_example C)

find_package(refimage 1.0 REQUIRED)

add_executable(stream stream.c)
target_link_libraries(stream PRIVATE refimage::refimage)

add_executable(stream_static stream.c)
target_link_libraries(stream_static PRIVATE refimage::refimage_static)
