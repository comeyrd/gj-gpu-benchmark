#ifndef O_DL_HPP  
#define  O_DL_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats dl_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class DLGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return dl_kernel(m,o);
    };
};

#endif
