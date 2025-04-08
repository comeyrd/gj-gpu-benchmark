cmake_minimum_required(VERSION 3.15)

project(gj-cuda LANGUAGES CUDA CXX)

add_subdirectory(kernels)

add_executable(main main.cpp)

set_target_properties(main PROPERTIES CUDA_ARCHITECTURES 70)

target_link_libraries(main PRIVATE kernels)