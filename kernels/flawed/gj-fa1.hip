#include "gj-fa1.hpp"
#include <hip/hip_runtime.h>

__global__ void fa1_fixRow(double *matrix, int size, int rowId) {
    __shared__ double Ri[512];
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size * rowId + colId];
    Aii = matrix[size * rowId + rowId];
    __syncthreads();
    Ri[colId] = Ri[colId] / Aii;
    matrix[size * rowId + colId] = Ri[colId];
}

__global__ void fa1_myfixColumn(double *matrix, int size, int colId) {
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    __shared__ double ratio;

    if (row_x != colId && matrix[row_x * size + colId] != 0) {
        ratio = matrix[row_x * size + colId] / matrix[colId * size + colId];
        double val = matrix[row_x * size + col_x] - ratio * matrix[colId * size + col_x];
        matrix[row_x * size + col_x] = val;
    }
}

void fa1_kernel(GJ_Utils::GJ_Matrix* m, GJ_Utils::S_Matrix* o) {
    hipError_t e;
    double* matrix;

    e = hipMalloc(&matrix, m->cols * m->rows * sizeof(double));
    e = hipMemcpy(matrix, m->data, m->cols * m->rows * sizeof(double), hipMemcpyHostToDevice);

    for (int l = 0; l < m->rows; l++) {
        hipLaunchKernelGGL(fa1_fixRow, dim3(1), dim3(m->cols), 0, 0, matrix, m->cols, l);
        hipDeviceSynchronize();

        hipLaunchKernelGGL(fa1_myfixColumn, dim3(m->rows), dim3(m->cols), 0, 0, matrix, m->cols, l);
        hipDeviceSynchronize();
    }

    e = hipMemcpy(m->data, matrix, m->cols * m->rows * sizeof(double), hipMemcpyDeviceToHost);
    e = hipFree(matrix);

    GJ_Utils::S_Matrix s = m->get_right_side();
    double* inner_out = new double[s.rows * s.cols]();
    memcpy(inner_out, s.data, s.rows * s.cols * sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out, o_owns_mem, s.rows, s.cols);
}
