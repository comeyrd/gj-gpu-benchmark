#include "kernels_list.hpp"
#include "gj-reference.hpp"
#include "gj-flawed.hpp"
#include "hip-utils.hpp"

void check_hip_error(hipError_t error_code,const char* file, int line){
    if(error_code != hipSuccess){
        std::string msg = std::string("HIP Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }

}


void retreive_kernels(){
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



HipProfiling::HipProfiling(){
    CHECK_HIP(hipEventCreate(&start));
    CHECK_HIP(hipEventCreate(&stop));
};

HipProfiling::~HipProfiling(){
    CHECK_HIP(hipEventDestroy(start));
    CHECK_HIP(hipEventDestroy(stop));
};

void HipProfiling::begin(){
    CHECK_HIP(hipEventRecord(start));
}

ExecutionStats HipProfiling::end(){
    CHECK_HIP(hipEventRecord(stop));
    CHECK_HIP(hipEventSynchronize(stop));

    ExecutionStats stats;
    CHECK_HIP(hipEventElapsedTime(&stats.elapsed, start, stop));
    return stats;
}