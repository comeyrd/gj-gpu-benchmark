add_subdirectory(utils)


if(BUILD_WITH_CUDA)
    file(GLOB FLAWED_CUDA "${CMAKE_SOURCE_DIR}/kernels/flawed/*.cu")
    add_library(kernels STATIC cuda/cuda_utils.cu reference/gj-reference.cu  ${FLAWED_CUDA})
    target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/cuda)
    set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES 70)
elseif(BUILD_WITH_HIP)
    message(FATAL_ERROR "HIP NOT IMPLEMENTED")
endif()

target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/kernels/flawed ${CMAKE_SOURCE_DIR}/kernels/reference)


target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(kernels PUBLIC utils)
