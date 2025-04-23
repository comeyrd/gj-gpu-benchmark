#include "matrix.hpp"
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdio.h>
double ACCEPTED_MIN = 0.002;
using namespace GJ_Utils;


void Matrix::update_memory(double *ptr, int row, int col) {
    if (row * col != rows * cols || !data) {
        // Resize internal storage
        data = std::make_unique<double[]>(row * col);
    }
    std::memcpy(data.get(), ptr, sizeof(double) * row * col);
    rows = row;
    cols = col;
}

// TODO use <<
void Matrix::print() {
    for (int i = 0; i < cols * rows; i++) {
        if (i == 0) {
            std::cout << "[ ";
        }
        std::cout << std::setw(4) << std::setfill(' ') << std::setprecision(3) << this->data[i] << " ";
        if (i == cols * rows - 1) {
            std::cout << "]" << std::endl;
        } else if ((i + 1) % cols == 0 && i != 0) {
            std::cout << std::endl
                      << "  ";
        }
    }
}
// TODO Generate "good" matrices, that will provide a low conditionned matrix.
// To achieve that, put 1 on L diagonal, and between 1 and 2 on U diagonal
// Keep the off-diagonal values small
void S_Matrix::fill_random_L() {
    Random_Number_Gen *gen = Random_Number_Gen::instance();
    for (int i = 0; i < this->cols; i++) {
        for (int j = 0; j <= i; j++) {
            double random_nbr = gen->generate();
            if (i == j && random_nbr == 0) {
                random_nbr += 1;
            }
            this->data[i * this->cols + j] = random_nbr;
        }
    }
}

void S_Matrix::fill_random_U() {
    Random_Number_Gen *gen = Random_Number_Gen::instance();
    for (int i = 0; i < this->rows; i++) {
        for (int j = i; j < this->cols; j++) {
            double random_nbr = gen->generate();
            if (i == j && random_nbr == 0) {
                random_nbr += 1;
            }
            this->data[i * this->cols + j] = random_nbr;
        }
    }
}

S_Matrix S_Matrix::L_random(int size) {
    S_Matrix m = S_Matrix(size);
    m.fill_random_L();
    return m;
}

S_Matrix S_Matrix::U_random(int size) {
    S_Matrix m = S_Matrix(size);
    m.fill_random_U();
    return m;
}

void Matrix::to_csv(std::ostream &output) {
    output << std::fixed << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            output << this->data[i * this->rows + j];
            if (j < (this->cols - 1))
                output << " ,";
        }
        output << std::endl;
    }
    output << std::endl;
}

Matrix Matrix::from_csv(std::istream &in) {
    std::vector<double> values;
    std::string line;
    int cols = -1;
    int rows = 0;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string token;
        int current_cols = 0;

        while (std::getline(ss, token, ',')) {
            // Remove spaces before/after comma
            size_t start = token.find_first_not_of(" \t");
            size_t end = token.find_last_not_of(" \t");
            if (start == std::string::npos) continue;

            std::string clean = token.substr(start, end - start + 1);
            values.push_back(std::stod(clean));
            current_cols++;
        }

        if (cols == -1) cols = current_cols;
        else if (current_cols != cols) {
            throw std::runtime_error("Inconsistent number of columns in CSV");
        }

        rows++;
    }
    std::unique_ptr<double[]> data = std::make_unique<double[]>(values.size());
    std::copy(values.begin(), values.end(), data.get());
    Matrix m(std::move(data), rows, cols);
    return m;
}



S_Matrix S_Matrix::times(const S_Matrix *m2) {
    if (this->rows != m2->rows || this->cols != m2->cols) {
        throw std::invalid_argument("Matrices must be of the same size for multiplication.");
    }
    S_Matrix res = S_Matrix(this->rows);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->cols; j++) {
            for (int k = 0; k < this->cols; k++) {
                res.data[i * this->cols + j] += this->data[i * this->cols + k] * m2->data[k * this->cols + j];
            }
        }
    }
    return res;
}

double S_Matrix::mean_difference(const S_Matrix *m2){
    double mean_error = 0;
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            mean_error += std::abs(this->data[i * this->cols + j]-m2->data[i * this->cols + j]);
        }
    }
    mean_error /= (this->rows * this->cols);
    return mean_error;
}


double S_Matrix::is_inverse(const S_Matrix *inverse) {
    S_Matrix m = this->times(inverse);
    double mean_error = 0;
    for (int i = 0; i < this->rows; ++i) {
        for (int j = 0; j < this->cols; ++j) {
            double value = m.data[i * this->cols + j];
            if (i == j) {
                value -= 1;
            }
            if (value < 0) {
                value *= -1;
            }
            mean_error += value;
        }
    }
    mean_error /= (this->rows * this->cols);
    return mean_error;
}

GJ_Matrix::GJ_Matrix(std::unique_ptr<double[]> data_ptr, S_Matrix *matrix) : Matrix(std::move(data_ptr), matrix->rows, matrix->cols * 2) {
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

GJ_Matrix::GJ_Matrix(S_Matrix *matrix) : Matrix(matrix->rows, matrix->cols * 2) {
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

void GJ_Matrix::print() {
    for (int i = 0; i < this->rows * this->cols; i++) {
        if (i == 0) {
            std::cout << "[ ";
        }
        std::cout << std::setw(4) << std::setfill(' ') << std::setprecision(3) << this->data[i] << " ";

        if (i == this->rows * this->cols - 1) {
            std::cout << "]" << std::endl
                      << std::endl;
        } else if ((i + 1) % (this->cols) == 0 && i != 0) {
            std::cout << std::endl
                      << "  ";
        } else if ((i + 1) % (this->rows) == 0 && i != 0) {
            std::cout << " | ";
        }
    }
}

S_Matrix GJ_Matrix::get_right_side() {
    S_Matrix rs = S_Matrix(this->rows);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->rows; j++) {
            rs.data[i * this->rows + j] = this->data[i * this->cols + j + this->rows];
        }
    }
    return rs;
}

S_Matrix GJ_Matrix::get_left_side() {
    S_Matrix rs = S_Matrix(this->rows);
    for (int i = 0; i < this->rows; i++) {
        for (int j = 0; j < this->rows; j++) {
            rs.data[i * this->rows + j] = this->data[i * this->cols + j];
        }
    }
    return rs;
}

S_Matrix S_Matrix::Random_Invertible(int size) {
    S_Matrix l = S_Matrix::L_random(size);
    S_Matrix u = S_Matrix::U_random(size);
    return l.times(&u);
}
