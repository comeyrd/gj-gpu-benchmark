#include "main.h"

int main(int argc, char **argv) {
    retreive_kernels();

    argparse::ArgumentParser program("gaussjordan");
    bool all;
    bool list;
    int repetitions = 5;
    int matrix_size = 100;
    std::vector<std::string> kernel;

    auto &group = program.add_mutually_exclusive_group(true);

    group.add_argument("-a", "--all").store_into(all);
    group.add_argument("-k", "--kernel").nargs(argparse::nargs_pattern::at_least_one).store_into(kernel); // Todo maybe list of strings to do multiple kernels ?
    group.add_argument("-l", "--list-kernels").store_into(list);

    program.add_argument("--repetitions", "-r").store_into(repetitions);
    program.add_argument("--matrix-size", "-m").store_into(matrix_size);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (all) {
        Kernel_umap Kmap = KernelsManager::instance()->getKernels();

        KStats_umap ks = do_kernel(Kmap, matrix_size, repetitions);
        for (const auto &[name, stats] : ks) {
            std::cout << "Kernel " << name << " " << stats << std::endl;
        }
    } else if (list) {
        list_kernels();
    } else {
        Kernel_umap Kmap = find_kernel(kernel);
        KStats_umap ks = do_kernel(Kmap, matrix_size, repetitions);
        for (const auto &[name, stats] : ks) {
            std::cout << "Kernel " << name << " " << stats << std::endl;
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
        } catch (const std::exception error) {
            std::cerr << "Kernel '" << kernel_name << "' not found" << std::endl;
        }
    }

    return filtered;
}

KStats_umap do_kernel(Kernel_umap kernels, int matrix_size, int repetitions) {
    try {
        GJ_Utils::S_Matrix source = GJ_Utils::S_Matrix::Random_Invertible(matrix_size);
        GJ_Utils::GJ_Matrix gj(&source);
        GJ_Utils::S_Matrix out(matrix_size);

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
            KernelStats ks = {e_stat, mean_err, repetitions, matrix_size};
            v_e_stat[name] = ks;
        }
        return v_e_stat;
    } catch (const std::exception &error) {
        std::cerr << "Error running kernel" << std::endl;
        std::cerr << error.what() << std::endl;
        exit(1);
    }
}