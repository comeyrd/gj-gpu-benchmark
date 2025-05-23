#include "kernels_list.hpp"
#include "gj-reference.hpp"
#include "gj-flawed.hpp"
#include "cuda-utils.hpp"

void check_cuda_error(cudaError_t error_code,const char* file, int line){
    if(error_code != cudaSuccess){
        std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

void retreive_kernels(){
    cudaSetDevice(0);
    KernelsManager* km = KernelsManager::instance();
    km->registerKernel("BS", std::make_shared<ReferenceGaussJordan>());
    km->registerKernel("OP",std::make_shared<OPGaussJordan>());
    km->registerKernel("RM",std::make_shared<RMGaussJordan>());
    km->registerKernel("DA",std::make_shared<DAGaussJordan>());
    km->registerKernel("CP",std::make_shared<CPGaussJordan>());
    km->registerKernel("DL",std::make_shared<DLGaussJordan>());
    km->registerKernel("RL",std::make_shared<RLGaussJordan>());
    km->registerKernel("YL", std::make_shared<YLGaussJordan>());
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