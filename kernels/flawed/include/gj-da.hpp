#ifndef O_da_HPP  
#define  O_da_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats da_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class DAGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return da_kernel(m,o);
    };
};

#endif
