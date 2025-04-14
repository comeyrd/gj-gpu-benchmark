#include "gj-oc.hpp"


__global__ void oc_fixRow(double *matrix, int size,int rowId){
    double Ri;
    double Aii;

    int colId = threadIdx.x;
    Ri = matrix[size*rowId + colId];
    Aii = matrix[size*rowId + rowId];
    Ri = Ri/Aii;
    matrix[size*rowId+colId] = Ri;
}

__global__ void oc_myfixColumn(double *matrix, int size, int colId){
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    __shared__ double ratio;
    if(row_x!=colId && matrix[row_x*size + colId] != 0){
        ratio = matrix[row_x*size + colId] / matrix[colId*size + colId];
        double val = matrix[row_x*size + col_x] - ratio * matrix[colId*size+col_x];
        matrix[row_x*size +col_x] = val;
    }
}

ExecutionStats oc_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    cudaSetDevice(1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t e;
    double* matrix;
    cudaEventRecord(start);
    e = cudaMalloc(&matrix,m->cols*m->rows*sizeof(double));
    e = cudaMemcpy(matrix,m->data,m->cols*m->rows*sizeof(double),cudaMemcpyHostToDevice);

    for(int l=0;l<m->rows;l++){
        oc_fixRow<<<1,m->cols>>>(matrix,m->cols,l);
        cudaDeviceSynchronize();
        oc_myfixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l);
        cudaDeviceSynchronize();

    }
    
    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);

    e = cudaMemcpy(out_gj.data,matrix,out_gj.cols*out_gj.rows*sizeof(double),cudaMemcpyDeviceToHost);
    e = cudaFree(matrix);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    ExecutionStats stats;
    cudaEventElapsedTime(&stats.elapsed, start, stop);

    GJ_Utils::S_Matrix s = out_gj.get_right_side();
    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool oc_owns_mem = true;
    o->update_memory(inner_out,oc_owns_mem,s.rows,s.cols);

    return stats;
};


