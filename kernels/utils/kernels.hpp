#ifndef KERNELS_HPP  
#define  KERNELS_HPP
#include "matrix.hpp"
#include <mutex>
#include <map>
#include <memory>
#include <iostream>

class IGaussJordan{
    public: 
        virtual ~IGaussJordan() = default;
        virtual void inverse(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix *o) const = 0;
};


class KernelsManager{
    private:
        static std::mutex _mtx;
        static KernelsManager* _manager;
        std::map<std::string, std::shared_ptr<IGaussJordan>> _kernels;

        KernelsManager() = default;
    public:
        KernelsManager(const KernelsManager&) = delete;
        KernelsManager& operator=(const KernelsManager&) = delete;

        static KernelsManager* getManager(){
            std::cout<<"yay"<<std::endl;
            std::lock_guard<std::mutex> lock(_mtx);
            if (_manager == nullptr) {
                _manager = new KernelsManager();
            }
            return _manager;
        }
        
        const std::map<std::string,std::shared_ptr<IGaussJordan>> getKernels(){
            return _kernels;
        };

        void registerKernel(const std::string& name, std::shared_ptr<IGaussJordan> kernel){
            _kernels[name] = kernel;
        };

};

std::mutex KernelsManager::_mtx;
KernelsManager* KernelsManager::_manager = nullptr;

#endif