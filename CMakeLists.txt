cmake_minimum_required(VERSION 3.15)

project(gj-cuda LANGUAGES CUDA CXX)

add_executable(main gauss_jordan.cu gj_utils.cpp)

include_directories(${CMAKE_SOURCE_DIR}/include)
