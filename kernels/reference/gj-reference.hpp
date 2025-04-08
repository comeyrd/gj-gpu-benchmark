#ifndef GJ_REFERENCE_HPP  
#define  GJ_REFERENCE_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

void reference_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class ReferenceGaussJordan : public IGaussJordan{
    public:
    void inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        reference_kernel(m,o);
    };
    ReferenceGaussJordan(){
        KernelsManager::getManager()->registerKernel("ReferenceGaussJordan",std::make_shared<ReferenceGaussJordan>());
        std::cout<<"a"<<std::endl;
    }
};

#endif
