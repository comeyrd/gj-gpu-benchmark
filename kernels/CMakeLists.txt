add_subdirectory(utils)


add_library(reference STATIC reference/gj-reference.cu)

set_target_properties(reference PROPERTIES CUDA_ARCHITECTURES 70)
target_include_directories(reference PUBLIC ${CMAKE_SOURCE_DIR}/kernels/reference)
target_link_libraries(reference PUBLIC utils)


add_library(flawed STATIC flawed/gj-fa1.cu)
set_target_properties(flawed PROPERTIES CUDA_ARCHITECTURES 70)
target_include_directories(flawed PUBLIC ${CMAKE_SOURCE_DIR}/kernels/flawed)
target_link_libraries(flawed PUBLIC utils)



add_library(kernels INTERFACE)

target_link_libraries(kernels INTERFACE reference flawed)

target_include_directories(kernels INTERFACE ${CMAKE_SOURCE_DIR}/kernels)
