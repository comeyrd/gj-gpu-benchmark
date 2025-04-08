#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include <cmath>
#include "matrix.hpp"
#include "kernels.hpp"

#include <string.h>

int main(int argc, char** argv){
    std::cout << "Gauss Jordan on GPU" << std::endl;
    int N = 5;
    bool debug = false;
    if(argc == 3){
        N = std::atoi(argv[1]);
        if((strcmp(argv[2], "1") == 0)){
            debug = true;
        }
    }
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
    

    auto kernels = KernelsManager::getManager()->getKernels();
    for (const auto& [name, kernel] : kernels) {
        std::cout << "Registered kernel: " << name << std::endl;
    }
    
    return 0;
   
    if(debug){
        std::cout<<"Matrix after Gj: "<<std::endl;
        gjm1.print();
    }
    GJ_Utils::S_Matrix ls = gjm1.get_right_side();
    auto [inv,max_error] = ls.is_inverse(&m3);
    std::cout << "My method is inverse : " << inv << " With max error : "<< max_error <<std::endl;
    if(!inv && debug){
        GJ_Utils::S_Matrix invt = ls.times(&m3);
        invt.print();
    }
    return 0;
}
