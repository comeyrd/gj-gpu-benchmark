#ifndef GJ_UTILS_HPP  
#define  GJ_UTILS_HPP

namespace GJ_Utils{

    class Matrix {
        private:
        bool owns_data;
        public:
            float * data;
            int rows,cols;
            Matrix(int rows, int cols);
            Matrix(float* externalData, int rows, int cols): rows(rows), cols(cols), data(externalData), owns_data(false) {}
            ~Matrix();
            float& at(int row, int col);
            void print();
    };

    class S_Matrix : public Matrix {
        public:
            S_Matrix(int size) : Matrix(size, size) {}
            S_Matrix(float* external_data, int size) : Matrix(external_data,size, size) {}
            void fill_random_L();
            void fill_random_U();
            S_Matrix times(const S_Matrix *m2);
            bool is_inverse(const S_Matrix *inverse);
    };
    
    class GJ_Matrix : public Matrix {
        public:
            GJ_Matrix(S_Matrix* matrix);
            GJ_Matrix(float* allocated,S_Matrix* matrix);
            void print();
            S_Matrix get_right_side();
            S_Matrix get_left_side();
    };
}



#endif 
