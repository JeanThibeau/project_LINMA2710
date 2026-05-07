#include "matrix.hpp"
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

Matrix::Matrix(int rows, int cols)
    : rows(0), cols(0), data()
{
    if (rows < 0 || cols < 0)
    {
        throw std::invalid_argument("Matrix dimensions must be non-negative");
    }
    this->rows = rows;
    this->cols = cols;
    data.assign(rows * cols, 0.0);
}

Matrix::Matrix(const Matrix &other)
    : rows(other.rows), cols(other.cols), data(other.data)
{
}

int Matrix::numRows() const
{
    return rows;
}

int Matrix::numCols() const
{
    return cols;
}

double Matrix::get(int i, int j) const
{
    if (i < 0 || i >= rows || j < 0 || j >= cols)
    {
        throw std::out_of_range("Matrix index out of range");
    }
    return data[i * cols + j];
}

void Matrix::set(int i, int j, double value)
{
    if (i < 0 || i >= rows || j < 0 || j >= cols)
    {
        throw std::out_of_range("Matrix index out of range");
    }
    data[i * cols + j] = value;
}

void Matrix::fill(double value)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i)
    {
        data[i] = value;
    }
}

Matrix Matrix::operator+(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }

    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i)
    {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

Matrix Matrix::operator-(const Matrix &other) const
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }

    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i)
    {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Matrix Matrix::operator*(const Matrix &other) const
{
    if (cols != other.rows)
    {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }

    Matrix result(rows, other.cols);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < other.cols; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; ++k)
            {
                sum += data[i * cols + k] * other.data[k * other.cols + j];
            }
            result.data[i * other.cols + j] = sum;
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i)
    {
        result.data[i] = data[i] * scalar;
    }
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result.data[j * rows + i] = data[i * cols + j];
        }
    }
    return result;
}

Matrix Matrix::apply(const std::function<double(double)> &func) const
{
    Matrix result(rows, cols);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i)
    {
        result.data[i] = func(data[i]);
    }
    return result;
}

void Matrix::sub_mul(double scalar, const Matrix &other)
{
    if (rows != other.rows || cols != other.cols)
    {
        throw std::invalid_argument("Matrix dimensions must match for sub_mul");
    }

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i = 0; i < rows * cols; ++i)
    {
        data[i] -= scalar * other.data[i];
    }
}
