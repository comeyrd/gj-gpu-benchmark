#ifndef CUDA_UTILS_HPP  
#define  CUDA_UTILS_HPP

#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

void check_cuda_error(cudaError_t error_code,const char* file, int line);

#endif