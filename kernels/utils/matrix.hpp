#ifndef MATRIX_HPP  
#define  MATRIX_HPP
#include <tuple>
#include <mutex>
#include <random>
#include <memory>

namespace GJ_Utils{


    class Matrix {
        friend class S_Matrix; 
        public:
            int rows,cols;
            std::unique_ptr<double[]> data;
            Matrix() = default;
            Matrix(int rows, int cols): rows(rows), cols(cols), data(std::make_unique<double[]>(rows * cols)) {}
            Matrix(std::unique_ptr<double[]> data_ptr, int rows, int cols):  rows(rows), cols(cols),data(std::move(data_ptr)){}
            ~Matrix() = default;
            void print();
            void update_memory(double* ptr,int row,int col);
            void to_csv(std::ostream &output);
            static Matrix from_csv(std::istream &in);

            Matrix& operator=(const Matrix& other) {
                if (this != &other) {
                    rows = other.rows;
                    cols = other.cols;
                    data = std::make_unique<double[]>(other.rows * other.cols);
                    std::copy(other.data.get(), other.data.get() + (other.rows * other.cols), data.get());
                }
                return *this;
            }
            Matrix& operator=(Matrix&& other) noexcept {
                if (this != &other) {
                    rows = other.rows;
                    cols = other.cols;
                    data = std::move(other.data);
                    other.rows = 0;
                    other.cols = 0;
                }
            return *this;}
            Matrix(Matrix&&) = default;
    };

    class S_Matrix : public Matrix {
        public:
            S_Matrix() = default;
            S_Matrix(int size) : Matrix(size, size) {}
            S_Matrix(std::unique_ptr<double[]> data_ptr, int size) : Matrix(std::move(data_ptr),size, size) {}
            static S_Matrix L_random(int size);
            static S_Matrix U_random(int size);
            void fill_random_L();
            void fill_random_U();
            S_Matrix times(const S_Matrix *m2);
            double is_inverse(const S_Matrix *inverse);
            static S_Matrix Random_Invertible(int size);
            double mean_difference(const S_Matrix *m2);
            static S_Matrix from_csv(std::istream &in){
                Matrix base = Matrix::from_csv(in);
                if(base.rows != base.cols){
                    throw std::runtime_error("Input is not a square matrix");
                }
                S_Matrix result(std::move(base.data), base.rows);
                return result;
            };

    };
    
    class GJ_Matrix : public Matrix {
        public:
            GJ_Matrix(S_Matrix* matrix);
            GJ_Matrix(S_Matrix matrix);
            GJ_Matrix(std::unique_ptr<double[]> data_ptr,S_Matrix* matrix);
            GJ_Matrix(std::unique_ptr<double[]> data_ptr, int rows, int cols) : Matrix(std::move(data_ptr),rows,cols){}
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
