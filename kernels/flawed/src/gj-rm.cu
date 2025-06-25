#include "gj-rm.hpp"
#include "cuda-utils.hpp"
//Array with a fixed size, unused memory

__global__ void rm_fixRow(double *matrix, int size,int rowId){
    __shared__ double Ri[1024];//bug
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    if(colId == rowId)
        Aii = Ri[rowId];
    __syncthreads();
    Ri[colId] = Ri[colId] / Aii;
    matrix[size*rowId+colId] = Ri[colId];
}

__global__ void rm_fixColumn(double *matrix, int size, int colId){
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    __shared__ double ratio;
    if(row_x!=colId && matrix[row_x*size + colId] != 0){
        if(col_x == 0)
            ratio = matrix[row_x*size + colId] / matrix[colId*size + colId];
        __syncthreads();
        double val = matrix[row_x*size + col_x] - ratio * matrix[colId*size+col_x];
        matrix[row_x*size +col_x] = val;
    }
}

ExecutionStats rm_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    CudaProfiling prof;
    double* matrix;
    CHECK_CUDA(cudaMalloc(&matrix,m->cols*m->rows*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(matrix,m->data.get(),m->cols*m->rows*sizeof(double),cudaMemcpyHostToDevice));
    prof.begin();
    for(int l=0;l<m->rows;l++){
        rm_fixRow<<<1,m->cols>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
        rm_fixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
    }
    ExecutionStats stats = prof.end();

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);

    CHECK_CUDA(cudaMemcpy(out_gj.data.get(),matrix,out_gj.cols*out_gj.rows*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(matrix));

    GJ_Utils::S_Matrix s = out_gj.get_right_side();
    *o = out_gj.get_right_side();

    return stats;
};
REGISTER_KERNEL(RMGaussJordan)
