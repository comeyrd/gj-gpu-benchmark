#include "argparse/argparse.hpp"
#include "kernels.hpp"
#include "kernels_list.hpp"
#include "matrix.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <string.h>


struct KernelStats{
    ExecutionStats e_stats;
    double mean_err;
    int repetitions;
    int size;
};

inline std::ostream& operator<<(std::ostream& os,KernelStats k_stat) {
    os <<k_stat.e_stats<<" ~Error = "<<k_stat.mean_err<<" #Repetitions = "<<k_stat.repetitions<< " Matrix Size = "<<k_stat.size;
    return os;
}


void all_kernels();
std::shared_ptr<IGaussJordan> find_kernel(std::string kernel);
KernelStats do_kernel(std::shared_ptr<IGaussJordan> kernel_fn,int matrix_size, int repetitions);
void list_kernels();



