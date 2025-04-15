#ifndef O_OP_HPP  
#define  O_OP_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats op_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class OPGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return op_kernel(m,o);
    };
};

#endif
