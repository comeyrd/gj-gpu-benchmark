import os
import argparse
import sys
reference_file = "gj-reference"

def copy_and_replace_bug(bigram: str, template_dir, output_dir,file_extension,description=""):
    bigram_upper = bigram.upper()
    bigram_lower = bigram.lower()
    template_file = f"{template_dir}/{reference_file}{file_extension}"
    if not os.path.exists(template_file):
        print(f"Error: Template file '{template_file}' does not exist.")
        return

    with open(template_file, 'r') as f:
        content = f.read()

    # Replace identifiers
    content = content.replace("REFERENCE", bigram_upper)
    content = content.replace("reference", bigram_lower)
    content = content.replace("Reference", bigram_upper)
    if file_extension == ".cu":
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if '#include "cuda-utils.hpp"' in line:
                lines.insert(i + 1, "//"+description)
                break
        content = "\n".join(lines)
        output_dir = os.path.join(output_dir, "src")
    elif file_extension == ".hpp":
        output_dir = output_dir + "/include"
    else:
        print(f"Error: Unsupported file extension '{file_extension}'.")
        return

    # Save new file
    new_filename = os.path.join(output_dir, f"gj-{bigram_lower}{file_extension}")
    with open(new_filename, 'w') as f:
        f.write(content)

    print(f"Created: {new_filename}")

def add_gj_include(bigram: str, flawed_path):
    include_line = f'#include "gj-{bigram.lower()}.hpp"\n'

    if not os.path.exists(flawed_path):
        print(f"Error: '{flawed_path}' not found.")
        return

    with open(flawed_path, 'r') as f:
        lines = f.readlines()

    if include_line in lines:
        print(f"{include_line.strip()} already exists in {flawed_path}")
        return

    # Find the line before the closing #endif to insert above it
    for i in reversed(range(len(lines))):
        if lines[i].strip().startswith("#endif"):
            if i > 0 and lines[i - 1].strip() == "":
                lines.insert(i - 1, include_line)
            else:
                lines.insert(i, include_line)
            break

    with open(flawed_path, 'w') as f:
        f.writelines(lines)

    print(f"Added include for '{bigram}' to {flawed_path}")


def add_kernel_registration(bigram: str, utils_path="utils.cu"):
    bigram_upper = bigram.upper()
    reg_line = f'    km->registerKernel("{bigram_upper}", std::make_shared<{bigram_upper}GaussJordan>());\n'

    if not os.path.exists(utils_path):
        print(f"Error: '{utils_path}' not found.")
        return

    with open(utils_path, 'r') as f:
        lines = f.readlines()

    # Check if the registration line already exists
    if reg_line in lines:
        print(f"{reg_line.strip()} already exists in {utils_path}")
        return

    # Find the last registration line inside retreive_kernels()
    insert_index = None
    inside_func = False
    for i, line in enumerate(lines):
        if "retreive_kernels()" in line:
            inside_func = True
        elif inside_func and "}" in line:
            insert_index = i  # insert before the closing brace
            break
        elif inside_func and 'km->registerKernel' in line:
            insert_index = i + 1  # keep updating to last known reg line
    if insert_index is not None:
        lines.insert(insert_index, reg_line)
        with open(utils_path, 'w') as f:
            f.writelines(lines)
        print(f"Inserted kernel registration for '{bigram_upper}' in {utils_path}")
    else:
        print("Could not find where to insert the kernel registration.")




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process bigram argument")
    parser.add_argument("bigram", type=str, help="A 2-character bigram identifier")
    parser.add_argument("desc", type=str, nargs='+', help="The description of the bug")

    args = parser.parse_args()

    bigram = args.bigram
    if len(bigram) != 2:
        print("Error: Bigram must be exactly 2 characters long.")
        sys.exit(1)
    description = " ".join(args.desc)    

    template_dir = "../kernels/reference"
    output_dir = "../kernels/flawed"
    copy_and_replace_bug(bigram,template_dir,output_dir,".hpp")
    copy_and_replace_bug(bigram,template_dir,output_dir,".cu",description)
    flawed_dir = "../kernels/flawed/include/gj-flawed.hpp"
    add_gj_include(bigram,flawed_dir)
    cuda_utils_dir = "../kernels/cuda/cuda-utils.cu"
    add_kernel_registration(bigram,cuda_utils_dir)

