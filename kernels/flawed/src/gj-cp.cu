#include "gj-cp.hpp"
#include "cuda-utils.hpp"
//Underparallelization
__global__ void cp_fixRow(double *matrix, int size,int rowId){
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

__global__ void cp_fixColumn(double *matrix, int n_col, int colId,int n_row ){
    int col_x = threadIdx.x;
    double ratio;
    for(int row_x = 0;row_x<n_row;row_x++){
        if(row_x!=colId && matrix[row_x*n_col + colId] != 0){
            if(col_x == 0)
                ratio = matrix[row_x*n_col + colId] / matrix[colId*n_col + colId];
            __syncthreads();
            double val = matrix[row_x*n_col + col_x] - ratio * matrix[colId*n_col+col_x];
            matrix[row_x*n_col +col_x] = val;
        }
}
}

ExecutionStats cp_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    CudaProfiling prof;

    double* matrix;
    CHECK_CUDA(cudaMalloc(&matrix,m->cols*m->rows*sizeof(double)));
    CHECK_CUDA(cudaMemcpy(matrix,m->data.get(),m->cols*m->rows*sizeof(double),cudaMemcpyHostToDevice));
    prof.begin();
    for(int l=0;l<m->rows;l++){
        cp_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
        cp_fixColumn<<<1,m->cols>>>(matrix,m->cols,l,m->rows);//bug
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
REGISTER_KERNEL(CPGaussJordan)
