#include "gj-reference.hpp"
#define CHECK_CUDA(error) check_cuda_error(error, __FILE__, __LINE__)

__global__ void reference_fixRow(double *matrix, int size,int rowId){
    extern __shared__ double Ri[];
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    __syncthreads();
    Aii = Ri[rowId];
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

void check_cuda_error(cudaError_t error_code,const char* file, int line){
    if(error_code != cudaSuccess){
        std::string msg = std::string("CUDA Error : ") + cudaGetErrorString(error_code) + std::string(" in : ") + file + std::string(" line ") + std::to_string(line);
        throw std::runtime_error(msg);
    }
}

ExecutionStats reference_kernel(GJ_Utils::GJ_Matrix* m,GJ_Utils::S_Matrix* o){
    cudaSetDevice(1);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t e;
    double* matrix;
    cudaEventRecord(start);
    e = cudaMalloc(&matrix,m->cols*m->rows*sizeof(double));
    CHECK_CUDA(e);
    e = cudaMemcpy(matrix,m->data,m->cols*m->rows*sizeof(double),cudaMemcpyHostToDevice);
    CHECK_CUDA(e);

    for(int l=0;l<m->rows;l++){
        reference_fixRow<<<1,m->cols,m->cols*sizeof(double)>>>(matrix,m->cols,l);
        e = cudaGetLastError();
        CHECK_CUDA(e);
        reference_myfixColumn<<<m->rows,m->cols>>>(matrix,m->cols,l);
        e = cudaGetLastError();
        CHECK_CUDA(e);
    }

    e = cudaDeviceSynchronize();
    CHECK_CUDA(e);

    GJ_Utils::GJ_Matrix out_gj = GJ_Utils::GJ_Matrix(m->rows);

    e = cudaMemcpy(out_gj.data,matrix,out_gj.cols*out_gj.rows*sizeof(double),cudaMemcpyDeviceToHost);
    CHECK_CUDA(e);
    e = cudaFree(matrix);
    CHECK_CUDA(e);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    ExecutionStats stats;
    cudaEventElapsedTime(&stats.elapsed, start, stop);

    GJ_Utils::S_Matrix s = out_gj.get_right_side();

    double* inner_out =  new double[s.rows * s.cols]();
    memcpy(inner_out,s.data,s.rows*s.cols*sizeof(double));
    bool o_owns_mem = true;
    o->update_memory(inner_out,o_owns_mem,s.rows,s.cols);

    return stats;
};