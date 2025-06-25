#include "gj-reference.hpp"
#include "gj-flawed.hpp"
#include "hip-utils.hpp"

void check_hip_error(hipError_t error_code,const char* file, int line){
    if(error_code != hipSuccess){
        std::string msg = std::string("HIP Error : ") + hipGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }

}

void setup_gpu() {
    hipSetDevice(0);
}

void reset_state() {
    hipDeviceReset();
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