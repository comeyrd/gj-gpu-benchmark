#ifndef O_RC_HPP  
#define  O_RC_HPP
#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats rc_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class RCGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return rc_kernel(m,o);
    };
};

#endif
