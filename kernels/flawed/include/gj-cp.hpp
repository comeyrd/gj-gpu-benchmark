#ifndef O_CP_HPP  
#define  O_CP_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats cp_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class CPGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return cp_kernel(m,o);
    };
};

#endif
