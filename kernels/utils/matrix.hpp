#ifndef MATRIX_HPP  
#define  MATRIX_HPP
#include <tuple>
#include <mutex>
#include <random>

namespace GJ_Utils{


    class Matrix {
        private:
        bool owns_data;
        public:
            double * data;
            int rows,cols;
            Matrix(int rows, int cols);
            Matrix(double* externalData, int rows, int cols): rows(rows), cols(cols), data(externalData), owns_data(false) {}
            ~Matrix();
            double& at(int row, int col);
            void print();
            void update_memory(double* ptr,bool owns,int row,int col);
    };

    class S_Matrix : public Matrix {
        public:
            S_Matrix(int size) : Matrix(size, size) {}
            S_Matrix(double* external_data, int size) : Matrix(external_data,size, size) {}
            void fill_random_L();
            void fill_random_U();
            S_Matrix times(const S_Matrix *m2);
            double is_inverse(const S_Matrix *inverse);
            
    };
    
    class GJ_Matrix : public Matrix {
        public:
            GJ_Matrix(S_Matrix* matrix);
            GJ_Matrix(double* allocated,S_Matrix* matrix);
            GJ_Matrix(double* allocated, int rows, int cols) : Matrix(allocated,rows,cols){}
            GJ_Matrix(int rows): Matrix(rows,rows*2){};
            void print();
            S_Matrix get_right_side();
            S_Matrix get_left_side();
    };

    class Random_Number_Gen{
        private:
            std::default_random_engine gen;  
            std::uniform_int_distribution<> dis;;
            static constexpr int LOWER_LIMIT = -100;
            static constexpr int UPPER_LIMIT = 100;
            Random_Number_Gen() : gen(std::random_device{}()),dis(LOWER_LIMIT, UPPER_LIMIT){}
        public:
        Random_Number_Gen(const Random_Number_Gen&) = delete;//Deleting copy
        Random_Number_Gen& operator=(const Random_Number_Gen&) = delete;

        static Random_Number_Gen* instance(){
            static Random_Number_Gen instance;
            return &instance;
        }

        int generate(){
            return dis(gen);
        }
    };
}




#endif 
