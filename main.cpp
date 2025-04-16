#include "main.h"

int main(int argc, char **argv) {
    retreive_kernels();

    argparse::ArgumentParser program("gaussjordan");
    bool all;
    bool list;
    int repetitions = 3;
    int matrix_size = 50;
    std::string kernel;

    auto &group = program.add_mutually_exclusive_group(true);

    group.add_argument("-a", "--all").store_into(all);
    group.add_argument("-k", "--kernel").store_into(kernel);
    group.add_argument("-l", "--list-kernels").store_into(list);

    program.add_argument("--repetitions", "-r").store_into(repetitions);
    program.add_argument("--matrix-size", "-ms").store_into(matrix_size);

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    if (all) {
       //To Implement
    } else if (list) {
        list_kernels();
    } else {
        std::shared_ptr<IGaussJordan> kernel_f = find_kernel(kernel);
        KernelStats ks = do_kernel(kernel_f, matrix_size, repetitions);
        std::cout << ks << std::endl;
    }

    return 0;
}

void list_kernels() {
    Kernel_umap kernels = KernelsManager::instance()->getKernels();
    for (const auto &[name, kernel] : kernels) {
        std::cout << name << std::endl;
    }
}

std::shared_ptr<IGaussJordan> find_kernel(std::string kernel) {
    Kernel_umap kernels = KernelsManager::instance()->getKernels();
    try {
        std::shared_ptr<IGaussJordan> k = kernels.at(kernel);
        std::cout << "Running Kernel " << kernel << std::endl;
        return k;
    } catch (const std::exception error) {
        std::cerr << "Kernel '" << kernel << "' not found" << std::endl;
        exit(1);
    }
}

KernelStats do_kernel(std::shared_ptr<IGaussJordan> kernel_fn, int matrix_size, int repetitions) {
    try {
        GJ_Utils::S_Matrix source = GJ_Utils::S_Matrix::Random_Invertible(matrix_size);
        GJ_Utils::GJ_Matrix gj(&source);
        GJ_Utils::S_Matrix out(matrix_size);

        ExecutionStats e_stat;
        double mean_err = 0;
        for (int r = 0; r < repetitions; r++) {
            e_stat += kernel_fn->inverse(&gj, &out);
            mean_err += out.is_inverse(&source);
        }

        e_stat = e_stat / repetitions;
        mean_err = mean_err / repetitions;
        KernelStats ks = {e_stat, mean_err, repetitions, matrix_size};
        return ks;
    } catch (const std::exception &error) {
        std::cerr << "Error running kernel" << std::endl;
        std::cerr << error.what() << std::endl;
        exit(1);
    }
}