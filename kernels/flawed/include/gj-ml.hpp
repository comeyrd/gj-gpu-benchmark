#ifndef O_ML_HPP  
#define  O_ML_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats ml_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class MLGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return ml_kernel(m,o);
    };
};

#endif
