#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include "matrix.hpp"
#include "kernels.hpp"
#include <fstream>

#include <string.h>
#include "kernels_list.hpp"

int main(int argc, char** argv){
    retreive_kernels();
    std::cout << "Gauss Jordan on GPU" << std::endl;
    //int N = 1000;
    bool debug = false;
    for (int N = 25;N<500; N = N*2){
        GJ_Utils::S_Matrix m1 = GJ_Utils::S_Matrix(N);
        m1.fill_random_U();
        GJ_Utils::S_Matrix m2 = GJ_Utils::S_Matrix(N);
        m2.fill_random_L();
        GJ_Utils::S_Matrix m3 = m2.times(&m1);
        if(debug){
            std::cout<<"Base matrix : "<<std::endl;
            m3.print();
        }
        std::ofstream outstream("matrix.txt");
        m3.to_csv(outstream);
        outstream.close();

        GJ_Utils::GJ_Matrix gjm1 =  GJ_Utils::GJ_Matrix(&m3);
        Kernel_umap kernels = KernelsManager::instance()->getKernels();
        GJ_Utils::S_Matrix o = GJ_Utils::S_Matrix(gjm1.rows);
        for (const auto& [name, kernel] : kernels) {
            double mean_err = 0;
            ExecutionStats e_stat;
            int n_runs = 20;
            for(int _ = 0;_<n_runs;_++){
                e_stat += kernel->inverse(&gjm1,&o);
                mean_err += o.is_inverse(&m3);
            }
            e_stat = e_stat / n_runs;
            mean_err = mean_err / n_runs;

            std::cout<<"N : "<<N<<" | " << name << " "<<mean_err <<" | " << e_stat << std::endl;
        }
        std::cout<<std::endl;
    }

    return 0;
}
