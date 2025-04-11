#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include "matrix.hpp"
#include "kernels.hpp"

#include <string.h>
#include "kernels_list.hpp"

int main(int argc, char** argv){
    retreive_kernels();
    std::cout << "Gauss Jordan on GPU" << std::endl;
    int N = 500;
    bool debug = false;
    if(argc == 3){
        N = std::atoi(argv[1]);
        if((strcmp(argv[2], "1") == 0)){
            debug = true;
        }
    }
    for (int m = 0;m<5;m++){
        GJ_Utils::S_Matrix m1 = GJ_Utils::S_Matrix(N);
        m1.fill_random_U();
        GJ_Utils::S_Matrix m2 = GJ_Utils::S_Matrix(N);
        m2.fill_random_L();
        GJ_Utils::S_Matrix m3 = m2.times(&m1);
        if(debug){
            std::cout<<"Base matrix : "<<std::endl;
            m3.print();
        }
        GJ_Utils::GJ_Matrix gjm1 =  GJ_Utils::GJ_Matrix(&m3);
        
        Kernel_umap kernels = KernelsManager::instance()->getKernels();
        GJ_Utils::S_Matrix o = GJ_Utils::S_Matrix(gjm1.rows);
        
        for (const auto& [name, kernel] : kernels) {
            double mean_err = 0;
            ExecutionStats e_stat;
            int n_runs = 10;
            for(int _ = 0;_<n_runs;_++){
                e_stat+=kernel->inverse(&gjm1,&o);
                mean_err+= o.is_inverse(&m3);
            }
            e_stat /=n_runs;
            mean_err /=n_runs;
            std::cout<<"Kernel " << name << " with mean error : " << mean_err << " and Execution Stats : " << e_stat <<  std::endl;
        }
    }

    return 0;
}
