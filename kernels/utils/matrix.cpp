#include "matrix.hpp"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <cstring>
double ACCEPTED_MIN = 0.002;
using namespace GJ_Utils;


Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    owns_data = true;
    data = new double[rows * cols]();  // Allocate and initialize to zero
}

Matrix::~Matrix() {
    if(owns_data)
        delete[] data;
}
void Matrix::update_memory(double* ptr,bool owns,int row,int col){
    if(owns_data)
        delete[] data;
    data = ptr;
    owns_data = owns;
    rows = row;
    cols = col;
}

double& Matrix::at(int row, int col){
    return data[row * cols + col]; 
}
//TODO use <<
void Matrix::print(){
    for(int i=0;i<cols*rows;i++){
        if(i == 0){
            std::cout << "[ ";
        }
        std::cout << std::setw(4) << std::setfill(' ') << std::setprecision(3)<< data[i] << " ";
        if(i==cols*rows-1){
            std::cout << "]" << std::endl;
        }
        else if((i+1)%cols==0 && i!=0){
            std::cout << std::endl<< "  ";
        }
} 
}
//TODO Generate "good" matrices, that will provide a low conditionned matrix.
//To achieve that, put 1 on L diagonal, and between 1 and 2 on U diagonal
//Keep the off-diagonal values small
void S_Matrix::fill_random_L(){
    Random_Number_Gen* gen = Random_Number_Gen::instance();
    for(int i=0;i<this->cols;i++){
        for(int j=0;j<=i;j++){
            double random_nbr = gen->generate();
            if (i == j&&random_nbr==0){
                random_nbr +=1;
            }
            this->data[i*this->cols + j] = random_nbr;
        }
    }
}

void S_Matrix::fill_random_U(){
    Random_Number_Gen* gen = Random_Number_Gen::instance();
    for(int i=0;i < this->rows;i++){
        for(int j=i ;j<this->cols;j++){
            double random_nbr = gen->generate();
            if (i == j&&random_nbr==0){
                random_nbr +=1;
            }
            this->data[i * this->cols + j] =random_nbr;
        }
    }
}

S_Matrix S_Matrix::L_random(int size){
    S_Matrix m = S_Matrix(size);
    m.fill_random_L();
    return m;
}

S_Matrix S_Matrix::U_random(int size){
    S_Matrix m = S_Matrix(size);
    m.fill_random_U();
    return m;
}

void Matrix::to_csv(std::ostream &output){
    output << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    for(int i=0;i<this->rows;i++){
        for(int j=0; j<this->cols;j++){
            output << this->data[i*this->rows+j];
            if(j< (this->cols - 1))
                output << " ,";
        }
        output << std::endl;
    }
    output << std::endl;
}


S_Matrix S_Matrix::times(const S_Matrix* m2){
    if (this->rows != m2->rows || this->cols != m2->cols) {
        throw std::invalid_argument("Matrices must be of the same size for multiplication.");
    }
    S_Matrix res = S_Matrix(this->rows);
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            res.data[i*this->rows +j] = 0; 
            for (int k = 0; k < this->cols; ++k) {
                res.data[i*this->rows+j] += this->data[i*this->rows+k] * m2->data[k*this->rows+j];
            }
        }
    }
    return res;
}

double S_Matrix::is_inverse(const S_Matrix *inverse){
    S_Matrix m = this->times(inverse);
    double mean_error = 0;
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            double value = m.data[i*this->cols+j];
            if (i == j) {
              value -= 1;
            }
            if(value<0){
                value*=-1;
            }
            mean_error+=value;
        }
    }
    mean_error /= (this->rows*this->cols);
    return mean_error;
}

GJ_Matrix::GJ_Matrix(double* allocated,S_Matrix* matrix) : Matrix(allocated,matrix->rows, matrix->cols * 2) {
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            this->data[i * this->cols + j] = matrix->data[i * matrix->cols + j];
        }
    }

    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = matrix->cols; j < this->cols; ++j) {
            if (i == (j - matrix->cols)) {
                this->data[i * this->cols + j] = 1;  
            } else {
                this->data[i * this->cols + j] = 0; 
            }
        }
    }
}

GJ_Matrix::GJ_Matrix(S_Matrix* matrix) : Matrix(matrix->rows, matrix->cols * 2) {
    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = 0; j < matrix->cols; ++j) {
            this->data[i * this->cols + j] = matrix->data[i * matrix->cols + j];
        }
    }

    for (int i = 0; i < matrix->rows; ++i) {
        for (int j = matrix->cols; j < this->cols; ++j) {
            if (i == (j - matrix->cols)) {
                this->data[i * this->cols + j] = 1;  
            } else {
                this->data[i * this->cols + j] = 0; 
            }
        }
    }
}
GJ_Matrix::GJ_Matrix(S_Matrix matrix) : Matrix(matrix.rows, matrix.cols * 2) {
    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = 0; j < matrix.cols; ++j) {
            this->data[i * this->cols + j] = matrix.data[i * matrix.cols + j];
        }
    }

    for (int i = 0; i < matrix.rows; ++i) {
        for (int j = matrix.cols; j < this->cols; ++j) {
            if (i == (j - matrix.cols)) {
                this->data[i * this->cols + j] = 1;  
            } else {
                this->data[i * this->cols + j] = 0; 
            }
        }
    }
}


void GJ_Matrix::print(){
    for(int i=0;i<this->rows*this->cols;i++){
        if(i == 0){
            std::cout << "[ ";
        }
        std::cout << std::setw(4) << std::setfill(' ') << std::setprecision(3)<< this->data[i] << " ";
        
        if(i == this->rows*this->cols-1){
            std::cout << "]" << std::endl<<std::endl;
        }
        else if((i+1)%(this->cols)==0 && i!=0){
            std::cout << std::endl<< "  ";
        }else if((i+1)%(this->rows)==0 && i!=0){
            std::cout << " | ";
        }
    } 
}


S_Matrix GJ_Matrix::get_right_side(){
    S_Matrix rs = S_Matrix(this->rows);
    for(int i=0;i<this->rows;i++){
        for(int j = 0; j < this->rows ; j++){
            rs.data[i*this->rows +j] = this->data[i*this->cols + j+this->rows];
        }
    }
    return rs;
}

S_Matrix GJ_Matrix::get_left_side(){
    S_Matrix rs = S_Matrix(this->rows);
    for(int i=0;i<this->rows;i++){
        for(int j=0;j<this->rows;j++){
            rs.data[i * this->rows + j] = this->data[i * this->cols + j];
        }
    }
    return rs;
}


S_Matrix S_Matrix::Random_Invertible(int size){
    S_Matrix l = S_Matrix::L_random(size);
    S_Matrix u  = S_Matrix::U_random(size);
    return l.times(&u);
}

