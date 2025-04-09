#ifndef GJ_FA1_HPP  
#define  GJ_FA1_HPP

#include "kernels.hpp"
#include <map>
#include <iostream>

void fa1_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o);

class Fa1GaussJordan : public IGaussJordan{
    public:
    void inverse(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) const override {
        fa1_kernel(m,o);
    };
};

#endif
