#include "gj_utils.hpp"
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <random>

int LOWER_LIMIT = -5;
int UPPER_LIMIT = 5;
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

double& Matrix::at(int row, int col){
    return data[row * cols + col]; 
}

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

void S_Matrix::fill_random_L(){
    std::default_random_engine gen;  
    std::uniform_int_distribution<> dis(LOWER_LIMIT, UPPER_LIMIT);
    int random_number = dis(gen);
    for(int i=0;i<this->cols;i++){
        for(int j=0;j<=i;j++){
            double random_nbr = dis(gen);
            if (i == j&&random_nbr==0){
                random_nbr +=1;
            }
            this->data[i*this->cols + j] = random_nbr;
        }
    }
}

void S_Matrix::fill_random_U(){
    std::default_random_engine gen;  
    std::uniform_int_distribution<> dis(LOWER_LIMIT, UPPER_LIMIT);
    int random_number = dis(gen);
    for(int i=0;i < this->rows;i++){
        for(int j=i ;j<this->cols;j++){
            double random_nbr = dis(gen);
            if (i == j&&random_nbr==0){
                random_nbr +=1;
            }
            this->data[i * this->cols + j] =random_nbr;
        }
    }
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

bool S_Matrix::is_inverse(const S_Matrix *inverse){
    S_Matrix m = this->times(inverse);
    bool ret = true;
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            if (i == j) {
                if(m.data[i*this->cols+j]>= 1+ACCEPTED_MIN || m.data[i*this->cols+j] <= 1-ACCEPTED_MIN  ){
                    std::cout<< m.data[i*this->cols+j] << "supposed to be 1"<<std::endl;
                    ret = false;
                }
            } else {
                if(m.data[i*this->cols+j]>= ACCEPTED_MIN ||m.data[i*this->cols+j] <= -ACCEPTED_MIN ){
                    std::cout<< m.data[i*this->cols+j] << "supposed to be 0"<<std::endl;
                    ret = false;
                }
            }
        }
    }
    return ret;
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

