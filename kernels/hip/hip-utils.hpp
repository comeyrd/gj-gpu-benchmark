#ifndef HIP_UTILS_HPP  
#define  HIP_UTILS_HPP

#include "hip/hip_runtime.h"
#include "kernels.hpp"

#define CHECK_HIP(error) check_hip_error(error, __FILE__, __LINE__)

void check_hip_error(hipError_t error_code,const char* file, int line);

class HipProfiling{
    private:
    hipEvent_t start;
    hipEvent_t stop;

    public:
    HipProfiling();
    ~HipProfiling();
    void begin();
    ExecutionStats end(); 
};



#endif