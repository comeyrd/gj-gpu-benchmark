#include "kernels_list.hpp"
#include "gj-reference.hpp"
#include "gj-flawed.hpp"


void check_cuda_error(cudaError_t error_code,const char* file, int line){
    if(error_code != cudaSuccess){
        std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

void retreive_kernels(){
    cudaSetDevice(0);
    KernelsManager* km = KernelsManager::instance();
    km->registerKernel("Reference", std::make_shared<ReferenceGaussJordan>());
    km->registerKernel("OC",std::make_shared<OCGaussJordan>());
    km->registerKernel("RC",std::make_shared<RCGaussJordan>());
    km->registerKernel("DA",std::make_shared<DAGaussJordan>());
}

