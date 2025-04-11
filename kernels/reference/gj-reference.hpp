#ifndef GJ_REFERENCE_HPP  
#define  GJ_REFERENCE_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats reference_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class ReferenceGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return reference_kernel(m,o);
    };
};

#endif
