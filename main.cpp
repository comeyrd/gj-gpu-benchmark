#include "main.h"

int main(int argc, char **argv) {
    argparse::ArgumentParser program("gaussjordan", VERSION_STRING);
    bool all;
    bool list;
    bool reuse;
    int repetitions = 5;
    int matrix_size = 100;
    std::vector<std::string> kernel;
    std::string stats_file_path = DEFAULT_STATS_FILE;

    auto &group = program.add_mutually_exclusive_group(true);

    group.add_argument("-a", "--all").store_into(all);
    group.add_argument("-k", "--kernel").nargs(argparse::nargs_pattern::at_least_one).store_into(kernel);
    group.add_argument("-l", "--list-kernels").store_into(list);

    program.add_argument("--repetitions", "-r").store_into(repetitions);
    program.add_argument("--matrix-size", "-m").store_into(matrix_size);
    program.add_argument("--reuse").store_into(reuse);

    program.add_argument("--stats-file").help("Enables the export of kernel execution stats to a JSON file with the specified PATH").default_value(DEFAULT_STATS_FILE).store_into(stats_file_path);
    
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if(list){
        list_kernels();
    }else{
        setup_gpu();
        Kernel_umap Kmap;
        if(all){
            Kmap = KernelsManager::instance()->getKernels();
        }else if(program.is_used("-k")){
            Kmap = find_kernel(kernel);
        }
        KStats_umap ks = do_kernel(Kmap, matrix_size, repetitions, reuse);
        std::cout<<ks;
        if(program.is_used("--stats-file")){
            json_to_file(ks,stats_file_path);
        }
    }

    return 0;
}

void list_kernels() {
    Kernel_umap kernels = KernelsManager::instance()->getKernels();
    for (const auto &[name, kernel] : kernels) {
        std::cout << name << std::endl;
    }
}

Kernel_umap find_kernel(std::vector<std::string> kernel_name_list) {
    Kernel_umap kernels_map = KernelsManager::instance()->getKernels();
    Kernel_umap filtered;

    for (std::string kernel_name : kernel_name_list) {
        try {
            filtered[kernel_name] = kernels_map.at(kernel_name);
        } catch (const std::exception &error) {
            std::cerr << "Kernel '" << kernel_name << "' not found" << std::endl;
        }
    }

    return filtered;
}

KStats_umap do_kernel(Kernel_umap kernels, int matrix_size, int repetitions, bool reuse) {
    try {
        GJ_Utils::S_Matrix source = GJ_Utils::S_Matrix::Random_Invertible(matrix_size);
        bool save = false;
        if (reuse) {
            std::ifstream file(DEFAULT_MATRIX_FILE);
            if (file) {
                GJ_Utils::S_Matrix from_file = GJ_Utils::S_Matrix::from_csv(file);
                if(from_file.cols == matrix_size){
                    std::cout << "Reusing matrix"<<std::endl;
                    source = from_file;
                }else{
                    save = true;
                }
            }else{
                save = true;
            }
            if(save){
                file.close();
                std::ofstream o_file(DEFAULT_MATRIX_FILE);
                source.to_csv(o_file);
            }
        }

        GJ_Utils::GJ_Matrix gj(&source);
        GJ_Utils::S_Matrix out(source.cols);

        KStats_umap v_e_stat;
        for (const auto &[name, k_func] : kernels) {
            std::cout << "Running Kernel " << name << std::endl;
            ExecutionStats e_stat;
            double mean_err = 0;
            for (int r = 0; r < repetitions; r++) {
                e_stat += k_func->inverse(&gj, &out);
                mean_err += out.is_inverse(&source);
            }
            e_stat = e_stat / repetitions;
            mean_err = mean_err / repetitions;
            KernelStats ks = {e_stat, mean_err, repetitions, source.cols};
            v_e_stat[name] = ks;
            reset_state();
        }
        return v_e_stat;
    } catch (const std::exception &error) {
        std::cerr << "Error running kernel" << std::endl;
        std::cerr << error.what() << std::endl;
        exit(1);
    }
}