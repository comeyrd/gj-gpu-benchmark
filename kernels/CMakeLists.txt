add_subdirectory(utils)


if(BUILD_WITH_CUDA)
    file(GLOB FLAWED_CUDA "${CMAKE_SOURCE_DIR}/kernels/flawed/src/*.cu")
    add_library(kernels STATIC cuda/cuda-utils.cu reference/gj-reference.cu  ${FLAWED_CUDA})
    target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/kernels/cuda)
    
    set_target_properties(kernels PROPERTIES CUDA_ARCHITECTURES 70)

elseif(BUILD_WITH_HIP)
    file(GLOB FLAWED_CUDA "${CMAKE_SOURCE_DIR}/kernels/flawed/src/*.hip")
    add_library(kernels STATIC hip/hip-utils.hip reference/gj-reference.hip  ${FLAWED_CUDA})
    target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/kernels/hip)

endif()

target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/kernels/flawed/include ${CMAKE_SOURCE_DIR}/kernels/reference)


target_include_directories(kernels PUBLIC ${CMAKE_SOURCE_DIR}/kernels)
target_link_libraries(kernels PUBLIC utils)
