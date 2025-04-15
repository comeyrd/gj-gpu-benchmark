#ifndef GJ_FLAWED_HPP  
#define  GJ_FLAWED_HPP

#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

void check_cuda_error(cudaError_t error_code,const char* file, int line);

#endif