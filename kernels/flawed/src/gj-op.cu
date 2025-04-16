#include "gj-op.hpp"
#include "cuda-utils.hpp"

//Useless synchronisation between host and device

__global__ void op_fixRow(double *matrix, int size,int rowId){
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

__global__ void op_fixColumn(double *matrix, int size, int colId){
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

ExecutionStats op_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    CudaProfiling prof;
    cudaError_t e;
    double* matrix;
    CHECK_CUDA(cudaMalloc(&matrix,m->cols*m->rows*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(matrix,m->data,m->cols*m->rows*sizeof(double),cudaMemcpyHostToDevice));
    
    prof.begin();
    for(int l=0;l<m->rows;l++){
        op_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
        e = cudaDeviceSynchronize();//bug
        CHECK_CUDA(e);
        op_fixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
        e = cudaDeviceSynchronize();//bug
        CHECK_CUDA(e);
    }
    ExecutionStats stats = prof.end();

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);

    CHECK_CUDA(cudaMemcpy(out_gj.data,matrix,out_gj.cols*out_gj.rows*sizeof(double),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(matrix));

    GJ_Utils::S_Matrix s = out_gj.get_right_side();

    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out,o_owns_mem,s.rows,s.cols);

    return stats;
};