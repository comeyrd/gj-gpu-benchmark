#include "kernels_list.hpp"
#include "gj-reference.hpp"
#include "gj-flawed.hpp"


void retreive_kernels(){
    KernelsManager* km = KernelsManager::instance();
    km->registerKernel("Reference", std::make_shared<ReferenceGaussJordan>());
    km->registerKernel("Optim",std::make_shared<OGaussJordan>());
}

