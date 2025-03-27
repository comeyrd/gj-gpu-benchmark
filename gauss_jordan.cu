#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include "gj_utils.hpp"



__global__ void fixRow(double *matrix, int size,int rowId){
    __shared__ double Ri[512];
    __shared__ double Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    Aii = matrix[size*rowId + rowId];
    __syncthreads();//Block synchronisation barrier
    Ri[colId] = Ri[colId]/Aii;
    matrix[size*rowId+colId] = Ri[colId];
}

__global__ void fixColumn(double *matrix, int size, int colId){
    int i = threadIdx.x;
    int j = blockIdx.x;
    __shared__ double col[512];
    __shared__ double colj[512];
    __shared__ double AColIdj;
    col[i] = matrix[i * size + colId];
    if(col[i]!=0){
        colj[i] = matrix[i * size + colId];
        AColIdj = matrix[colId * size + j];
        if (i!= colId){
            colj[i] = colj[i] - AColIdj * col[i];
        }
        matrix[i * size + j] = colj[i];
    }
}

__global__ void myfixColumn(double *matrix, int size, int colId){
    int col_x = threadIdx.x;
    int row_x = blockIdx.x;
    __shared__ double ratio;
    if(row_x!=colId && matrix[row_x*size + colId] != 0){
        ratio = matrix[row_x*size + colId] / matrix[colId*size + colId];
        double val = matrix[row_x*size + col_x] - ratio * matrix[colId*size+col_x];
        matrix[row_x*size +col_x] = val;
    }
}

int main(int argc, char** argv){
    std::cout << "Gauss Jordan on GPU" << std::endl;
    int N = 10;

    GJ_Utils::S_Matrix m1 = GJ_Utils::S_Matrix(N);
    m1.fill_random_U();
    GJ_Utils::S_Matrix m2 = GJ_Utils::S_Matrix(N);
    m2.fill_random_L();
    GJ_Utils::S_Matrix m3 = m2.times(&m1);
    m3.print();

    double *matrix;
    cudaMallocManaged(&matrix, N*(N+N)*sizeof(double));

    GJ_Utils::GJ_Matrix gjm1 =  GJ_Utils::GJ_Matrix(matrix,&m3);
    //gjm1.print();

    
    int col = N*2;
    int row = N;
    for(int l=0;l<row;l++){
        fixRow<<<1,col>>>(matrix,col,l);
        cudaDeviceSynchronize();
        //gjm1.print();
        myfixColumn<<<row,col>>>(matrix,col,l);
        cudaDeviceSynchronize();
        //gjm1.print();
    }
    gjm1.print();
    GJ_Utils::S_Matrix ls = gjm1.get_right_side(); 
    auto [inv,max_error] = ls.is_inverse(&m3);
    std::cout << "My method is inverse : " << inv << " With max error : "<< max_error <<std::endl;
    if(!inv){
        GJ_Utils::S_Matrix invt = ls.times(&m3);
        invt.print();
    }
    
    return 0;
} 