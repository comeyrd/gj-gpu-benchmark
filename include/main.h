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
#include "version.h"

struct KernelStats {
    ExecutionStats e_stats;
    double mean_err;
    int repetitions;
    int size;
};

std::string DEFAULT_MATRIX_FILE = "matrix.csv";
typedef std::unordered_map<std::string, KernelStats> KStats_umap;

inline std::ostream &operator<<(std::ostream &os, KernelStats k_stat) {
    os <<  k_stat.e_stats << " ~Error = " << k_stat.mean_err << " #Repetitions = " << k_stat.repetitions << " Matrix Size = " << k_stat.size;
    return os;
}

Kernel_umap find_kernel(std::vector<std::string> kernel_name_list);
KStats_umap do_kernel(Kernel_umap kernels, int matrix_size, int repetitions,bool reuse);
void list_kernels();
