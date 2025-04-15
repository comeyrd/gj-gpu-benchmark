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
    km->registerKernel("Reference", std::make_shared<ReferenceGaussJordan>());
    km->registerKernel("OP",std::make_shared<OPGaussJordan>());
    km->registerKernel("RC",std::make_shared<RCGaussJordan>());
    km->registerKernel("DA",std::make_shared<DAGaussJordan>());
}
