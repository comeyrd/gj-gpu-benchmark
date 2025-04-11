#ifndef O_FA1_HPP  
#define  O_FA1_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats o_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class OGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return o_kernel(m,o);
    };
};

#endif
