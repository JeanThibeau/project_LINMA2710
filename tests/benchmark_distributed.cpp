#include "distributed_matrix.hpp"
#include "matrix.hpp"

#include <mpi.h>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <vector>

static Matrix make_matrix(int rows, int cols, double seed)
{
    Matrix matrix(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix.set(i, j, seed + i * 0.01 + j * 0.001);
        }
    }
    return matrix;
}

static double time_multiply_transposed(const DistributedMatrix& left, const DistributedMatrix& right)
{
    auto start = std::chrono::high_resolution_clock::now();
    Matrix result = left.multiplyTransposed(right);
    auto end = std::chrono::high_resolution_clock::now();
    (void)result;
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        std::cout << "size,rank_count,left_rows,right_rows,local_time_ms" << std::endl;
    }

    std::vector<int> sizes = {64, 128, 256, 512};

    for (int n : sizes)
    {
        Matrix left = make_matrix(n, n, 1.0);
        Matrix right = make_matrix(n, n, 2.0);

        DistributedMatrix leftDist(left, size);
        DistributedMatrix rightDist(right, size);

        // Warm-up
        (void)leftDist.multiplyTransposed(rightDist);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_ms = time_multiply_transposed(leftDist, rightDist);

        double max_t = 0.0;
        MPI_Reduce(&t_ms, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            std::cout << n << "," << size << "," << n << "," << n << "," << std::fixed << std::setprecision(3)
                      << max_t << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}