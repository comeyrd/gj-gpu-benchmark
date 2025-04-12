#ifndef KERNEL_LIST_HPP  
#define  KERNEL_LIST_HPP

#include "kernels.hpp"

struct RegisterKernel {
    RegisterKernel(const std::string& name, std::shared_ptr<IGaussJordan> kernel) {
        KernelsManager::instance()->registerKernel(name, kernel);
    }
};

void retreive_kernels();


#endif
