import os
import subprocess
import sys
from new_bugg import add_kernel_registration

def run_hipify(input_file):
    # Call hipify-perl and capture its output
    result = subprocess.run(["hipify-perl", input_file], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"hipify-perl failed for {input_file}:\n{result.stderr}")
        return None
    return result.stdout

def modify_hipified_code(code):
    lines = code.splitlines()
    modified_lines = []

    for i, line in enumerate(lines):
        if i == 0 and line.strip() == '#include "hip/hip_runtime.h"':
            continue  # Skip first line if it's the HIP runtime include
        line = line.replace('#include "cuda-utils.hpp"', '#include "hip-utils.hpp"')
        line = line.replace('CudaProfiling', 'HipProfiling')
        line = line.replace('CHECK_CUDA', 'CHECK_HIP')
        modified_lines.append(line)

    return "\n".join(modified_lines) + "\n"

def convert_cuda_files_in_folder(folder_path,include_path,recode=False):
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".cu"):
                file_exists = False
                input_path = os.path.join(root, filename)
                output_path = os.path.splitext(input_path)[0] + ".hip"

                # Skip if .hip file already exists
                if os.path.exists(output_path):
                    print(f"Skipping (already exists): {output_path}")
                    if recode:
                        file_exists = True
                    else:
                        continue

                print(f"Converting: {input_path} -> {output_path}")

                hipified_code = run_hipify(input_path)
                if hipified_code is None:
                    continue

                modified_code = modify_hipified_code(hipified_code)

                with open(output_path, "w") as out_file:
                    out_file.write(modified_code)

                bigram = filename.split("-")[-1].replace(".cu", "")
                if not file_exists:
                    add_kernel_registration(bigram, include_path)

if __name__ == "__main__":
    folder = "../kernels/flawed/src/"
    include = "../kernels/hip/hip-utils.hip"
    convert_cuda_files_in_folder(folder,include,True)

    #folder = "../kernels/reference/"
    #convert_cuda_files_in_folder(folder)