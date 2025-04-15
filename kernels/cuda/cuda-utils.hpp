#ifndef CUDA_UTILS_HPP  
#define  CUDA_UTILS_HPP

#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

void check_cuda_error(cudaError_t error_code,const char* file, int line);

class CudaProfiling{
    private:
    cudaEvent_t start;
    cudaEvent_t stop;

    public:
    CudaProfiling();
    ~CudaProfiling();
    void begin();
    ExecutionStats end(); 
};

//TODO Update the tools to use "nsys" and "ncu" and other.

#endif