#include "gj-reference.hpp"
#include "gj-flawed.hpp"
#include "cuda-utils.hpp"

void check_cuda_error(cudaError_t error_code,const char* file, int line){
    if(error_code != cudaSuccess){
        std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

void setup_gpu(){
    cudaSetDevice(0);
}

void reset_state(){
    cudaDeviceReset();
}

CudaProfiling::CudaProfiling(){
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
};

CudaProfiling::~CudaProfiling(){
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
};

void CudaProfiling::begin(){
    CHECK_CUDA(cudaEventRecord(start));
}

ExecutionStats CudaProfiling::end(){
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    ExecutionStats stats;
    CHECK_CUDA(cudaEventElapsedTime(&stats.elapsed, start, stop));
    return stats;
}