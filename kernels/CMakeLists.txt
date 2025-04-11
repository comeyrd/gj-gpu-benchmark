add_subdirectory(utils)

if(BUILD_WITH_CUDA)
    add_library(reference STATIC reference/gj-reference.cu)
    set_target_properties(reference PROPERTIES CUDA_ARCHITECTURES 70)
elseif(BUILD_WITH_HIP)
    add_library(reference STATIC reference/gj-reference.hip)
endif()

target_include_directories(reference PUBLIC ${CMAKE_SOURCE_DIR}/kernels/reference)
target_link_libraries(reference PUBLIC utils)

if(BUILD_WITH_CUDA)
    file(GLOB FLAWED_CUDA "${CMAKE_SOURCE_DIR}/kernels/flawed/*.cu")
    add_library(flawed STATIC ${FLAWED_CUDA})
    set_target_properties(flawed PROPERTIES CUDA_ARCHITECTURES 70)
elseif(BUILD_WITH_HIP)
    file(GLOB FLAWED_HIP "${CMAKE_SOURCE_DIR}/kernels/flawed/*.hip")
    add_library(flawed STATIC flawed/gj-fa1.hip)
endif()

target_include_directories(flawed PUBLIC ${CMAKE_SOURCE_DIR}/kernels/flawed)
target_link_libraries(flawed PUBLIC utils)

add_library(kernels INTERFACE)

target_link_libraries(kernels INTERFACE reference flawed)

target_include_directories(kernels INTERFACE ${CMAKE_SOURCE_DIR}/kernels)
