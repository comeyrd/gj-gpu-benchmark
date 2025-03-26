#include <iostream>
#include <math.h>
#include <iomanip>
#include <stdio.h>
#include "gj_utils.hpp"


__global__ void fixRow(float *matrix, int size,int rowId){
    int colId = threadIdx.x;
    matrix[size*rowId + colId] = matrix[size*rowId + colId] / matrix[size*rowId + rowId];
}

__global__ void fixColumn(float *matrix, int wideness,int current_col){
    int i = threadIdx.x; // What row
    int j = blockIdx.x; //What column
    if(i != current_col){
        matrix[i*wideness + j] = matrix[i*wideness + j] - (matrix[i*wideness + current_col] / matrix[current_col * wideness + current_col]) * matrix[current_col*wideness + j];
}
}

int main(int argc, char** argv){
    std::cout << "Gauss Jordan on GPU" << std::endl;

    GJ_Utils::S_Matrix m1 = GJ_Utils::S_Matrix(3);
    m1.fill_random_U();
    GJ_Utils::S_Matrix m2 = GJ_Utils::S_Matrix(3);
    m2.fill_random_L();
    GJ_Utils::S_Matrix m3 = m2.times(&m1);
    m3.print();

    int N = 3;
    float *matrix;
    cudaMallocManaged(&matrix, N*(N+N)*sizeof(float));

    GJ_Utils::GJ_Matrix gjm1 =  GJ_Utils::GJ_Matrix(matrix,&m3);
    gjm1.print();

    
    int col = N*2;
    int row = N;
    for(int l=0;l<row;l++){
        gjm1.print();
        fixRow<<<1,col>>>(matrix,col,l);
        cudaDeviceSynchronize();
        gjm1.print();
        fixColumn<<<row,col>>>(matrix,col,l);
        cudaDeviceSynchronize();
    }
    gjm1.print();
    GJ_Utils::S_Matrix ls = gjm1.get_right_side(); 
    std::cout << "is_inverse : " << ls.is_inverse(&m3) <<std::endl;


    float result[9] = {331.0f / 400, 51.0f / 50, 9.0f / 100, 
        51.0f / 50, 34.0f / 25, 3.0f / 25, 
        9.0f / 100, 3.0f / 25, 1.0f / 25};

    GJ_Utils::S_Matrix m_result = GJ_Utils::S_Matrix(result,3);

    std::cout << "is_inverse : " << m_result.is_inverse(&m3) <<std::endl;



    return 0;
} 