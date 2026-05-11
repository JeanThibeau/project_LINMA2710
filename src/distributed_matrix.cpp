#include "distributed_matrix.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>

// The matrix is split by columns across MPI processes.
// Each process stores a local Matrix with a subset of columns.
// Columns are distributed as evenly as possible.

DistributedMatrix::DistributedMatrix(const Matrix& matrix, int numProcs)
    : globalRows(matrix.numRows()),
      globalCols(matrix.numCols()),
      localCols(0),
      startCol(0),
      numProcesses(numProcs),
      rank(0),
      localData(matrix.numRows(), 1)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (numProcesses <= 0)
    {
        throw std::invalid_argument("Number of MPI processes must be positive");
    }

    const int baseCols = globalCols / numProcesses;
    const int remainder = globalCols % numProcesses;

    localCols = baseCols + (rank < remainder ? 1 : 0);
    startCol = rank * baseCols + std::min(rank, remainder);

    localData = Matrix(globalRows, localCols);
    for (int i = 0; i < globalRows; ++i)
    {
        for (int localJ = 0; localJ < localCols; ++localJ)
        {
            localData.set(i, localJ, matrix.get(i, startCol + localJ));
        }
    }
}

DistributedMatrix::DistributedMatrix(const DistributedMatrix& other)
    : globalRows(other.globalRows),
      globalCols(other.globalCols),
      localCols(other.localCols),
      startCol(other.startCol),
      numProcesses(other.numProcesses),
      rank(other.rank),
      localData(other.localData)
{
}

int DistributedMatrix::numRows() const { return globalRows; }
int DistributedMatrix::numCols() const { return globalCols; }
const Matrix& DistributedMatrix::getLocalData() const { return localData; }

double DistributedMatrix::get(int i, int j) const
{
    if (i < 0 || i >= globalRows || j < 0 || j >= globalCols)
    {
        throw std::out_of_range("DistributedMatrix index out of range");
    }

    const int localJ = localColIndex(j);
    if (localJ < 0)
    {
        throw std::out_of_range("Requested column is not stored on this MPI rank");
    }

    return localData.get(i, localJ);
}

void DistributedMatrix::set(int i, int j, double value)
{
    if (i < 0 || i >= globalRows || j < 0 || j >= globalCols)
    {
        throw std::out_of_range("DistributedMatrix index out of range");
    }

    const int localJ = localColIndex(j);
    if (localJ < 0)
    {
        throw std::out_of_range("Requested column is not stored on this MPI rank");
    }

    localData.set(i, localJ, value);
}

int DistributedMatrix::globalColIndex(int localColIdx) const
{
    if (localColIdx < 0 || localColIdx >= localCols)
    {
        throw std::out_of_range("Local column index out of range");
    }
    return startCol + localColIdx;
}

int DistributedMatrix::localColIndex(int globalColIdx) const
{
    if (globalColIdx < startCol || globalColIdx >= startCol + localCols)
    {
        return -1;
    }
    return globalColIdx - startCol;
}

int DistributedMatrix::ownerProcess(int globalColIdx) const
{
    if (globalColIdx < 0 || globalColIdx >= globalCols)
    {
        throw std::out_of_range("Global column index out of range");
    }

    const int baseCols = globalCols / numProcesses;
    const int remainder = globalCols % numProcesses;
    const int cutoff = (baseCols + 1) * remainder;

    if (globalColIdx < cutoff)
    {
        return globalColIdx / (baseCols + 1);
    }

    return remainder + (globalColIdx - cutoff) / baseCols;
}

void DistributedMatrix::fill(double value)
{
    localData.fill(value);
}

DistributedMatrix DistributedMatrix::operator+(const DistributedMatrix& other) const
{
    if (globalRows != other.globalRows ||
        globalCols != other.globalCols ||
        localCols != other.localCols ||
        startCol != other.startCol)
    {
        throw std::invalid_argument("DistributedMatrix dimensions/partitioning must match for addition");
    }

    Matrix dummy(globalRows, globalCols);
    DistributedMatrix result(dummy, numProcesses);
    result.localData = localData + other.localData;
    return result;
}

DistributedMatrix DistributedMatrix::operator-(const DistributedMatrix& other) const
{
    if (globalRows != other.globalRows ||
        globalCols != other.globalCols ||
        localCols != other.localCols ||
        startCol != other.startCol)
    {
        throw std::invalid_argument("DistributedMatrix dimensions/partitioning must match for subtraction");
    }

    Matrix dummy(globalRows, globalCols);
    DistributedMatrix result(dummy, numProcesses);
    result.localData = localData - other.localData;
    return result;
}

DistributedMatrix DistributedMatrix::operator*(double scalar) const
{
    Matrix dummy(globalRows, globalCols);
    DistributedMatrix result(dummy, numProcesses);
    result.localData = localData * scalar;
    return result;
}

Matrix DistributedMatrix::transpose() const
{
    return gather().transpose();
}

void DistributedMatrix::sub_mul(double scalar, const DistributedMatrix& other)
{
    if (globalRows != other.globalRows ||
        globalCols != other.globalCols ||
        localCols != other.localCols ||
        startCol != other.startCol)
    {
        throw std::invalid_argument("DistributedMatrix dimensions/partitioning must match for sub_mul");
    }

    localData.sub_mul(scalar, other.localData);
}

DistributedMatrix DistributedMatrix::apply(const std::function<double(double)>& func) const
{
    Matrix dummy(globalRows, globalCols);
    DistributedMatrix result(dummy, numProcesses);
    result.localData = localData.apply(func);
    return result;
}

DistributedMatrix DistributedMatrix::applyBinary(
    const DistributedMatrix& a,
    const DistributedMatrix& b,
    const std::function<double(double, double)>& func)
{
    if (a.globalRows != b.globalRows ||
        a.globalCols != b.globalCols ||
        a.localCols != b.localCols ||
        a.startCol != b.startCol)
    {
        throw std::invalid_argument("DistributedMatrix dimensions/partitioning must match for applyBinary");
    }

    Matrix dummy(a.globalRows, a.globalCols);
    DistributedMatrix result(dummy, a.numProcesses);
    for (int i = 0; i < a.globalRows; ++i)
    {
        for (int localJ = 0; localJ < a.localCols; ++localJ)
        {
            result.localData.set(
                i,
                localJ,
                func(a.localData.get(i, localJ), b.localData.get(i, localJ)));
        }
    }
    return result;
}

DistributedMatrix multiply(const Matrix& left, const DistributedMatrix& right)
{
    if (left.numCols() != right.globalRows)
    {
        throw std::invalid_argument("Matrix dimensions must match for multiplication");
    }

    Matrix dummy(left.numRows(), right.globalCols);
    DistributedMatrix result(dummy, right.numProcesses);

    for (int i = 0; i < left.numRows(); ++i)
    {
        for (int localJ = 0; localJ < right.localCols; ++localJ)
        {
            double sum = 0.0;
            for (int k = 0; k < left.numCols(); ++k)
            {
                sum += left.get(i, k) * right.localData.get(k, localJ);
            }
            result.localData.set(i, localJ, sum);
        }
    }

    return result;
}

Matrix DistributedMatrix::multiplyTransposed(const DistributedMatrix& other) const
{
    if (globalCols != other.globalCols ||
        localCols != other.localCols ||
        startCol != other.startCol)
    {
        throw std::invalid_argument("DistributedMatrix partitioning must match for multiplyTransposed");
    }

    // Local partial result: each rank computes its contribution
    Matrix localPartial(globalRows, other.globalRows);
    localPartial.fill(0.0);

    double t_start = 0.0, t_comp_end = 0.0, t_comm_end = 0.0;
    t_start = MPI_Wtime();

    // Local computation
    for (int i = 0; i < globalRows; ++i)
    {
        for (int j = 0; j < other.globalRows; ++j)
        {
            double sum = 0.0;
            for (int localK = 0; localK < localCols; ++localK)
            {
                sum += localData.get(i, localK) * other.localData.get(j, localK);
            }
            localPartial.set(i, j, sum);
        }
    }

    t_comp_end = MPI_Wtime();

    // Prepare send buffer and run the collective reduction; include packing in comm time
    const int resultSize = globalRows * other.globalRows;
    std::vector<double> sendbuf(resultSize, 0.0);
    std::vector<double> recvbuf(resultSize, 0.0);

    for (int i = 0; i < globalRows; ++i)
    {
        for (int j = 0; j < other.globalRows; ++j)
        {
            sendbuf[i * other.globalRows + j] = localPartial.get(i, j);
        }
    }

    double comm_start = MPI_Wtime();
    MPI_Allreduce(
        sendbuf.data(),
        recvbuf.data(),
        resultSize,
        MPI_DOUBLE,
        MPI_SUM,
        MPI_COMM_WORLD);
    t_comm_end = MPI_Wtime();

    Matrix result(globalRows, other.globalRows);
    for (int i = 0; i < globalRows; ++i)
    {
        for (int j = 0; j < other.globalRows; ++j)
        {
            result.set(i, j, recvbuf[i * other.globalRows + j]);
        }
    }

    double t_end = MPI_Wtime();

    // Compute timings (seconds)
    double t_comp = t_comp_end - t_start;
    double t_comm = t_comm_end - comm_start;
    double t_total = t_end - t_start;

    // Aggregate timings across ranks: report max and average
    double max_comp = 0.0, max_comm = 0.0, max_total = 0.0;
    double sum_comp = 0.0, sum_comm = 0.0, sum_total = 0.0;

    MPI_Reduce(&t_comp, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_total, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Reduce(&t_comp, &sum_comp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &sum_comm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_total, &sum_total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        int world_size = 1;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        double avg_comp = sum_comp / world_size;
        double avg_comm = sum_comm / world_size;
        double avg_total = sum_total / world_size;

        // Report sizes and timings (seconds)
        const long long bytes = static_cast<long long>(resultSize) * static_cast<long long>(sizeof(double));
        std::cout << "[multiplyTransposed] globalRows=" << globalRows
                  << " otherRows=" << other.globalRows
                  << " resultSize=" << resultSize
                  << " bytes(reduced)=" << bytes << std::endl;
        std::cout << "  comp (s): max=" << max_comp << " avg=" << avg_comp << std::endl;
        std::cout << "  comm (s): max=" << max_comm << " avg=" << avg_comm << std::endl;
        std::cout << "  total (s): max=" << max_total << " avg=" << avg_total << std::endl;
        double comm_frac = (avg_comm) / (avg_total > 0 ? avg_total : 1.0);
        std::cout << "  comm fraction (avg)=" << comm_frac << "\n" << std::flush;
    }

    return result;
}

double DistributedMatrix::sum() const
{
    double localSum = 0.0;
    for (int i = 0; i < globalRows; ++i)
    {
        for (int localJ = 0; localJ < localCols; ++localJ)
        {
            localSum += localData.get(i, localJ);
        }
    }

    double globalSum = 0.0;
    MPI_Allreduce(&localSum, &globalSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return globalSum;
}

Matrix DistributedMatrix::gather() const
{
    std::vector<int> counts(numProcesses, 0);
    std::vector<int> displs(numProcesses, 0);

    const int baseCols = globalCols / numProcesses;
    const int remainder = globalCols % numProcesses;

    for (int p = 0; p < numProcesses; ++p)
    {
        const int pLocalCols = baseCols + (p < remainder ? 1 : 0);
        counts[p] = globalRows * pLocalCols;
        if (p > 0)
        {
            displs[p] = displs[p - 1] + counts[p - 1];
        }
    }

    std::vector<double> sendbuf(globalRows * localCols, 0.0);
    for (int i = 0; i < globalRows; ++i)
    {
        for (int localJ = 0; localJ < localCols; ++localJ)
        {
            sendbuf[i * localCols + localJ] = localData.get(i, localJ);
        }
    }

    std::vector<double> recvbuf(globalRows * globalCols, 0.0);
    MPI_Allgatherv(
        sendbuf.data(),
        static_cast<int>(sendbuf.size()),
        MPI_DOUBLE,
        recvbuf.data(),
        counts.data(),
        displs.data(),
        MPI_DOUBLE,
        MPI_COMM_WORLD);

    Matrix gathered(globalRows, globalCols);
    for (int p = 0; p < numProcesses; ++p)
    {
        const int pLocalCols = baseCols + (p < remainder ? 1 : 0);
        const int pStartCol = p * baseCols + std::min(p, remainder);
        const int pOffset = displs[p];

        for (int i = 0; i < globalRows; ++i)
        {
            for (int localJ = 0; localJ < pLocalCols; ++localJ)
            {
                gathered.set(
                    i,
                    pStartCol + localJ,
                    recvbuf[pOffset + i * pLocalCols + localJ]);
            }
        }
    }

    return gathered;
}

void sync_matrix(Matrix *matrix, int rank, int src)
{
    int rows = 0;
    int cols = 0;

    if (rank == src)
    {
        rows = matrix->numRows();
        cols = matrix->numCols();
    }

    MPI_Bcast(&rows, 1, MPI_INT, src, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, src, MPI_COMM_WORLD);

    std::vector<double> buffer(rows * cols, 0.0);
    if (rank == src)
    {
        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < cols; ++j)
            {
                buffer[i * cols + j] = matrix->get(i, j);
            }
        }
    }

    MPI_Bcast(buffer.data(), rows * cols, MPI_DOUBLE, src, MPI_COMM_WORLD);

    if (rank != src)
    {
        *matrix = Matrix(rows, cols);
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            matrix->set(i, j, buffer[i * cols + j]);
        }
    }
}
