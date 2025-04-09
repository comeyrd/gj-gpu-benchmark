cmake_minimum_required(VERSION 3.15)

project(gj-cuda LANGUAGES CXX)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    message(STATUS "Building with CUDA")
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

add_executable(main main.cpp)

target_link_libraries(main PRIVATE kernels)