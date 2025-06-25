#include "argparse/argparse.hpp"
#include "nlohmann/json.hpp"
#include "kernels.hpp"
#include "gpu-utils.hpp"
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

void to_json(nlohmann::json& j, const ExecutionStats& e) {
    j = nlohmann::json{{"elapsed", e.elapsed}};
}

void to_json(nlohmann::json& j, const KernelStats& k) {
    j = nlohmann::json{
        {"e_stats", k.e_stats},
        {"mean_err", k.mean_err},
        {"repetitions", k.repetitions},
        {"size", k.size}
    };
}

std::string DEFAULT_MATRIX_FILE = "matrix.csv";
std::string DEFAULT_STATS_FILE = "stats.json";

typedef std::unordered_map<std::string, KernelStats> KStats_umap;

void json_to_file(const KStats_umap& stats_map, const std::string& filename) {
    nlohmann::json j = stats_map;
    std::ofstream file(filename);
    if (file.is_open()) {
        file << j.dump(4); 
    } else {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
    }
}

inline std::ostream &operator<<(std::ostream &os, KernelStats k_stat) {
    os <<  k_stat.e_stats << " ~Error = " << k_stat.mean_err << " #Repetitions = " << k_stat.repetitions << " Matrix Size = " << k_stat.size;
    return os;
}

inline std::ostream &operator<<(std::ostream &os, KStats_umap ks){
    for (const auto &[name, stats] : ks) {
        os << "Kernel " << name << " " << stats << std::endl;
    }
    return os;
}

Kernel_umap find_kernel(std::vector<std::string> kernel_name_list);
KStats_umap do_kernel(Kernel_umap kernels, int matrix_size, int repetitions,bool reuse);
void list_kernels();
