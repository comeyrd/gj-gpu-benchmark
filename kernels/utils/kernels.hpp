#ifndef KERNELS_HPP  
#define  KERNELS_HPP
#include "matrix.hpp"
#include <mutex>
#include <unordered_map>
#include <memory>
#include <iostream>

struct ExecutionStats{
    float elapsed = 0;

ExecutionStats operator+(const ExecutionStats& other) const {
    return {
        this->elapsed + other.elapsed,
    };
}
ExecutionStats& operator+=(const ExecutionStats& other) {
    this->elapsed += other.elapsed;
    return *this;
}
ExecutionStats& operator/=(float scalar) {
    this->elapsed /= scalar;
    return *this;
}

ExecutionStats operator/(float scalar) const {
    return {
        this->elapsed / scalar,
    };
}
};
inline std::ostream& operator<<(std::ostream& os,ExecutionStats e_stat) {
    os << "elapsed = " << e_stat.elapsed << " ms";
    return os;
}


class IGaussJordan{
    public: 
        virtual ~IGaussJordan() = default;
        virtual ExecutionStats inverse(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix *o) const = 0;
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

#define REGISTER_KERNEL(ClassName) \
    namespace { \
        struct ClassName##AutoRegister { \
            ClassName##AutoRegister() { \
                KernelsManager::instance()->registerKernel(#ClassName, std::make_shared<ClassName>()); \
            } \
        }; \
        static ClassName##AutoRegister global_##ClassName##AutoRegister; \
    }

#endif