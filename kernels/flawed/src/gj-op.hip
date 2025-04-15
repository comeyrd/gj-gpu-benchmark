#include "hip/hip_runtime.h"
#include "gj-op.hpp"
#include "hip-utils.hpp"
//Useless synchronisation between host and device

__global__ void oc_fixRow(double *matrix, int size,int rowId){
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

__global__ void oc_fixColumn(double *matrix, int size, int colId){
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

ExecutionStats oc_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    hipSetDevice(1);
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipError_t e;
    double* matrix;
    hipEventRecord(start);
    e = hipMalloc(&matrix,m->cols*m->rows*sizeof(double));
    CHECK_HIP(e);
    e = hipMemcpy(matrix,m->data,m->cols*m->rows*sizeof(double),hipMemcpyHostToDevice);
    CHECK_HIP(e);

    for(int l=0;l<m->rows;l++){
        oc_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        e = hipGetLastError();
        CHECK_HIP(e);
        e = hipDeviceSynchronize();//bug
        CHECK_HIP(e);
        oc_fixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l);
        e = hipGetLastError();
        CHECK_HIP(e);
        e = hipDeviceSynchronize();//bug
        CHECK_HIP(e);
    }

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);

    e = hipMemcpy(out_gj.data,matrix,out_gj.cols*out_gj.rows*sizeof(double),hipMemcpyDeviceToHost);
    CHECK_HIP(e);
    e = hipFree(matrix);
    CHECK_HIP(e);

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    ExecutionStats stats;
    hipEventElapsedTime(&stats.elapsed, start, stop);

    GJ_Utils::S_Matrix s = out_gj.get_right_side();

    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out,o_owns_mem,s.rows,s.cols);

    return stats;
};