#include "kernels.hpp"
#include "gj-reference.hpp"
#include "gj-fa1.hpp"


struct RegisterKernel {
    RegisterKernel(const std::string& name, std::shared_ptr<IGaussJordan> kernel) {
        KernelsManager::instance()->registerKernel(name, kernel);
    }
};

void retreive_kernels(){
    KernelsManager* km = KernelsManager::instance();
    km->registerKernel("Reference", std::make_shared<ReferenceGaussJordan>());
    km->registerKernel("FA1", std::make_shared<Fa1GaussJordan>());
}

