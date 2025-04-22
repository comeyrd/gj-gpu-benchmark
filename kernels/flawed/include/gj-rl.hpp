#ifndef O_RL_HPP  
#define  O_RL_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

ExecutionStats rl_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class RLGaussJordan : public IGaussJordan{
    public:
    ExecutionStats inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        return rl_kernel(m,o);
    };
};

#endif
