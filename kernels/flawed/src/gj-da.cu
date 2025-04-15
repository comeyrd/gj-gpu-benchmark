#include "gj-da.hpp"
#include "cuda-utils.hpp"
//Global Memory for Shared Value
__global__ void da_fixRow(double *matrix, int size,int rowId){
    extern __shared__ double Ri[];
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    if(colId == rowId)
        Aii = Ri[rowId];
    __syncthreads();
    Ri[colId] = Ri[colId]/Aii;
    matrix[size*rowId+colId] = Ri[colId];
}

__global__ void da_fixColumn(double *matrix, int size, int colId,double *ratio/*bug*/){
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    if(row_x!=colId && matrix[row_x*size + colId] != 0){
        if(col_x == 0)
            ratio[row_x] = matrix[row_x*size + colId] / matrix[colId*size + colId];
        __syncthreads();
        double val = matrix[row_x*size + col_x] - ratio[row_x] * matrix[colId*size+col_x];
        matrix[row_x*size +col_x] = val;
    }
}

ExecutionStats da_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    CudaProfiling prof;
    cudaError_t e;
    double* matrix;
    e = cudaMalloc(&matrix,m->cols*m->rows*sizeof(double));
    CHECK_CUDA(e);
    e = cudaMemcpy(matrix,m->data,m->cols*m->rows*sizeof(double),cudaMemcpyHostToDevice);
    CHECK_CUDA(e);

    double* ratio;
    e = cudaMalloc(&ratio,m->rows*sizeof(double));
    CHECK_CUDA(e);

    prof.begin();
    for(int l=0;l<m->rows;l++){
        da_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        e = cudaGetLastError();
        CHECK_CUDA(e);
        da_fixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l,ratio);
        e = cudaGetLastError();
        CHECK_CUDA(e);
    }
    ExecutionStats stats = prof.end();

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);
    e = cudaMemcpy(out_gj.data,matrix,out_gj.cols*out_gj.rows*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK_CUDA(e);
    e = cudaFree(matrix);
    CHECK_CUDA(e);
    e = cudaFree(ratio);
    CHECK_CUDA(e);
    


    GJ_Utils::S_Matrix s = out_gj.get_right_side();

    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out,o_owns_mem,s.rows,s.cols);

    return stats;
};