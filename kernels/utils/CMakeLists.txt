add_library(utils STATIC matrix.cpp kernels.cpp)
target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR}/kernels/utils)