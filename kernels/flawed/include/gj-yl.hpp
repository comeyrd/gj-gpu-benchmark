#ifndef GJ_YL_HPP  
#define  GJ_YL_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats yl_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class YLGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return yl_kernel(m,o);
    };
};

#endif
