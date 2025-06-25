#include "gj-da.hpp"
#include "hip-utils.hpp"
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
    HipProfiling prof;
    double* matrix;
    CHECK_HIP(hipMalloc(&matrix,m->cols*m->rows*sizeof(double)));
    CHECK_HIP(hipMemcpy(matrix,m->data.get(),m->cols*m->rows*sizeof(double),hipMemcpyHostToDevice));

    double* ratio;
    CHECK_HIP(hipMalloc(&ratio,m->rows*sizeof(double)));

    prof.begin();
    for(int l=0;l<m->rows;l++){
        da_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        CHECK_HIP(hipGetLastError());
        da_fixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l,ratio);
        CHECK_HIP(hipGetLastError());
    }
    ExecutionStats stats = prof.end();

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);
    CHECK_HIP(hipMemcpy(out_gj.data.get(),matrix,out_gj.cols*out_gj.rows*sizeof(double),hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(matrix));
    CHECK_HIP(hipFree(ratio));
    


    GJ_Utils::S_Matrix s = out_gj.get_right_side();
    *o = out_gj.get_right_side();

    return stats;
};
REGISTER_KERNEL(DAGaussJordan)
