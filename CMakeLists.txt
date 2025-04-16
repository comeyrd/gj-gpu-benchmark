cmake_minimum_required(VERSION 3.15)

project(gj-cuda LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Using CUDA toolkit: ${CMAKE_CUDA_COMPILER}")
    set(BUILD_WITH_CUDA TRUE)
    set(BUILD_WITH_HIP FALSE)
    enable_language(CUDA)
else()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        set(BUILD_WITH_HIP TRUE)
        set(BUILD_WITH_CUDA FALSE)
        enable_language(HIP)
        message(STATUS "Building with HIP")
    else()
        message(FATAL_ERROR "CUDA and HIP not found, can't compile")
    endif()
endif()

add_subdirectory(kernels)

add_executable(gaussjordan main.cpp)


include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(argparse)

target_include_directories(gaussjordan PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(gaussjordan PRIVATE kernels argparse)