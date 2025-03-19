#include <iostream>
#include <math.h>
#include <iomanip>

void init_square_matrix(float* matrix, int size);
void display_square_matrix(float* matrix, int size);
void init_square_matrix_wId(float* matrix, int size);
void display_square_matrix_wId(float* matrix, int size);

__global__ void fixRow(float *matrix, int size,int rowId){
    __shared__ float Ri[512];
    __shared__ float Aii;

    int colId = threadIdx.x;
    Ri[colId] = matrix[size*rowId + colId];
    Aii = matrix[size*rowId + rowId];
    __syncthreads();//Block synchronisation barrier
    Ri[colId] = Ri[colId]/Aii;
    matrix[size*rowId+colId] = Ri[colId];
}



__global__ void finishColumn(float *matrix, int wideness,int current_col){
    int i = threadIdx.x; // What column
    int j = blockIdx.x; //What row

    
}

__global__ void fixColumn(float *matrix, int size, int colId){
    int i = threadIdx.x;
    int j = blockIdx.x;
    __shared__ float col[512];
    __shared__ float colj[512];
    __shared__ float AColIdj;
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

int main(int argc, char** argv){
    std::cout << "Gauss Jordan on GPU" << std::endl;
    int N = 3;
    float *matrix;
    cudaMallocManaged(&matrix, N*(N+N)*sizeof(float));
    init_square_matrix_wId(matrix,N);
    int row = N*2;
    int col = N;

    fixRow<<<1,row>>>(matrix,row,0);
    cudaDeviceSynchronize();
    //fixColumn<<<row,col>>>(matrix,row,0);
    cudaDeviceSynchronize();
    display_square_matrix_wId(matrix,N);
    return 0;
}


void init_square_matrix(float* matrix, int size){
    for(int i=0;i<size*size;i++){
        matrix[i] = i + 4;
    }   
}

void init_square_matrix_wId(float* matrix, int size){
    for(int i=0;i<size;i++){
        for(int k=0;k<size;k++){
            matrix[i*size*2 + k] = i + 4;
        }
    }   
    for(int s=0;s<size;s++){
        for(int m=0;m<size;m++){
            if(s == m)
                matrix[s*size*2+ m + size] = 1;
            else
                matrix[s*size*2+ m + size ] = 0;
        }
    }
}


void display_square_matrix(float* matrix, int size){
    for(int i=0;i<size*size;i++){
        if(i == 0){
            std::cout << "[ ";
        }
        std::cout << std::setw(4) << std::setfill(' ') << std::setprecision(3)<< matrix[i] << " ";
        if(i==size*size-1){
            std::cout << "]" << std::endl;
        }
        else if((i+1)%size==0 && i!=0){
            std::cout << std::endl<< "  ";
        }
    } 

} 

void display_square_matrix_wId(float* matrix, int size){
    for(int i=0;i<size*(size*2);i++){
        if(i == 0){
            std::cout << "[ ";
        }
        std::cout << std::setw(4) << std::setfill(' ') << std::setprecision(3)<< matrix[i] << " ";
        
        if(i==(size*2)*size-1){
            std::cout << "]" << std::endl;
        }
        else if((i+1)%(size*2)==0 && i!=0){
            std::cout << std::endl<< "  ";
        }else if((i+1)%(size)==0 && i!=0){
            std::cout << " | ";
        }
    } 

} 