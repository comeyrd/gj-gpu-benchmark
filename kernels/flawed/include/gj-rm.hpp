#ifndef O_RM_HPP  
#define  O_RM_HPP
#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats rm_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class RMGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return rm_kernel(m,o);
    };
};

#endif
