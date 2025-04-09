#ifndef KERNELS_HPP  
#define  KERNELS_HPP
#include "matrix.hpp"
#include <mutex>
#include <unordered_map>
#include <memory>
#include <iostream>

class IGaussJordan{
    public: 
        virtual ~IGaussJordan() = default;
        virtual void inverse(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix *o) const = 0;
};

typedef std::unordered_map<std::string, std::shared_ptr<IGaussJordan>> Kernel_umap;

class KernelsManager{
    private:
        Kernel_umap _kernels;
    public:
    
        static KernelsManager* instance(){
            static KernelsManager manager;
            return &manager;
        }
        const std::unordered_map<std::string,std::shared_ptr<IGaussJordan>> &getKernels(){
            return _kernels;
        };

        void registerKernel(const std::string& name, std::shared_ptr<IGaussJordan> kernel){
            _kernels[name] = kernel;
        };
        

};


#endif