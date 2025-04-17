#include "gj-dl.hpp"
#include "cuda-utils.hpp"
//Using a double** instead of Double*
__global__ void dl_fixRow(double **matrix, int size,int rowId){
    extern __shared__ double Ri[];
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[rowId][colId];
    if(colId == rowId)
        Aii = Ri[rowId];
    __syncthreads();
    Ri[colId] = Ri[colId]/Aii;
    matrix[rowId][colId] = Ri[colId];
}

__global__ void dl_fixColumn(double **matrix, int size, int colId){
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    __shared__ double ratio;
    if(row_x!=colId && matrix[row_x][colId] != 0){
        if(col_x == 0)
            ratio = matrix[row_x][colId] / matrix[colId][colId];
        __syncthreads();
        double val = matrix[row_x][col_x] - ratio * matrix[colId][col_x];
        matrix[row_x][col_x] = val;
    }
}

ExecutionStats dl_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    CudaProfiling prof;

    double** h_matrix = new double*[m->rows]; 
    for (int l=0;l<m->rows;l++){
        CHECK_CUDA(cudaMalloc(&h_matrix[l],m->cols*sizeof(double)));
        CHECK_CUDA(cudaMemcpy(h_matrix[l],m->data+(l*m->cols),m->cols*sizeof(double),cudaMemcpyHostToDevice));
    }
    double** matrix;
    CHECK_CUDA(cudaMalloc(&matrix, m->rows * sizeof(double*)));
    CHECK_CUDA(cudaMemcpy(matrix, h_matrix, m->rows * sizeof(double*), cudaMemcpyHostToDevice));
    delete[] h_matrix;
    prof.begin();
    for(int l=0;l<m->rows;l++){
        dl_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
        dl_fixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l);
        CHECK_CUDA(cudaGetLastError());
    }
    ExecutionStats stats = prof.end();

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);


    double** r_matrix = new double*[m->rows]; 
    CHECK_CUDA(cudaMemcpy(r_matrix, matrix, m->rows * sizeof(double*), cudaMemcpyDeviceToHost));

    for (int l=0;l<m->rows;l++){
        CHECK_CUDA(cudaMemcpy(out_gj.data+(l*m->cols),r_matrix[l],m->cols*sizeof(double),cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaFree(r_matrix[l]));
    }
    cudaFree(matrix);
    GJ_Utils::S_Matrix s = out_gj.get_right_side();

    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out,o_owns_mem,s.rows,s.cols);

    return stats;
};