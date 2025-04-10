#include "gj-reference.hpp"
#include <hip/hip_runtime.h>


__global__ void reference_fixRow(double *matrix, int size,int rowId){
    __shared__ double Ri[512];
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    Aii = matrix[size*rowId + rowId];
    Ri[colId] = Ri[colId]/Aii;
    matrix[size*rowId+colId] = Ri[colId];
}

__global__ void reference_myfixColumn(double *matrix, int size, int colId){
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    __shared__ double ratio;
    if(row_x!=colId && matrix[row_x*size + colId] != 0){
        ratio = matrix[row_x*size + colId] / matrix[colId*size + colId];
        double val = matrix[row_x*size + col_x] - ratio * matrix[colId*size+col_x];
        matrix[row_x*size +col_x] = val;
    }
}

void reference_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    double* matrix;
    hipMalloc(&matrix,m->cols*m->rows*sizeof(double));
    //TODO manage error and throwing
    hipMemcpy(matrix,m->data,m->cols*m->rows*sizeof(double),hipMemcpyHostToDevice);

    for(int l=0;l<m->rows;l++){
        hipLaunchKernelGGL(reference_fixRow, dim3(1), dim3(m->cols), 0, 0, matrix, m->cols, l);
        hipDeviceSynchronize();
        hipLaunchKernelGGL(reference_myfixColumn, dim3(m->rows), dim3(m->cols), 0, 0, matrix, m->cols, l);
        hipDeviceSynchronize();
    }

    hipMemcpy(m->data,matrix,m->cols*m->rows*sizeof(double),hipMemcpyDeviceToHost);
    hipFree(matrix);
    GJ_Utils::S_Matrix s = m->get_right_side();
    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out,o_owns_mem,s.rows,s.cols);
};
