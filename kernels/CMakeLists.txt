add_subdirectory(utils)


add_library(reference STATIC reference/gj-kernel.cu)

set_target_properties(reference PROPERTIES CUDA_ARCHITECTURES 70)
target_include_directories(reference PUBLIC ${CMAKE_SOURCE_DIR}/reference)

target_link_libraries(reference PUBLIC utils)

add_library(kernels INTERFACE)
target_link_libraries(kernels INTERFACE reference)
