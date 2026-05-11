#include "matrix_opencl.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <memory>

std::shared_ptr<KernelCache> MatrixCL::kernels_ = nullptr;

cl::Program loadAndBuildProgram(cl::Context context,
                                const std::vector<cl::Device>& devices,
                                const std::string& sourceCode,
                                const std::string& kernel_name_for_error)
{
    cl::Program program(context, sourceCode);
    try {
        program.build(devices);
    } catch (const cl::BuildError& err) {
        std::cerr << "OpenCL Build Error for kernel source '" << kernel_name_for_error << "':\n"
                  << err.what() << "(" << err.err() << ")" << std::endl;
        for (const auto& pair : err.getBuildLog()) {
            std::cerr << "Device " << pair.first.getInfo<CL_DEVICE_NAME>() << ":" << std::endl;
            std::cerr << pair.second << std::endl;
        }
        throw;
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL Error during program build for '" << kernel_name_for_error << "': "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    }
    return program;
}

// --- OpenCL Kernel Source Code ---

const std::string kernel_source_fill = R"(
    __kernel void fill(__global float* matrix, float value, int rows, int cols) {
        int idx = get_global_id(0);
        int total = rows * cols;
        if (idx < total) {
            matrix[idx] = value;
        }
    }
)";

const std::string kernel_source_add = R"(
    __kernel void add(__global const float* A,
                      __global const float* B,
                      __global float* C,
                      int rows, int cols) {
        int idx = get_global_id(0);
        int total = rows * cols;
        if (idx < total) {
            C[idx] = A[idx] + B[idx];
        }
    }
)";

const std::string kernel_source_sub_mul = R"(
    __kernel void sub_mul(__global float* A,
                          __global const float* B,
                          float scalar,
                          int rows, int cols) {
        int idx = get_global_id(0);
        int total = rows * cols;
        if (idx < total) {
            A[idx] -= scalar * B[idx];
        }
    }
)";

const std::string kernel_source_transpose = R"(
    __kernel void transpose(__global const float* A,
                            __global float* B,
                            int A_rows, int A_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < A_rows && j < A_cols) {
            B[j * A_rows + i] = A[i * A_cols + j];
        }
    }
)";

const std::string kernel_source_matrix_mul = R"(
    __kernel void matrix_mul(__global const float* A,
                             __global const float* B,
                             __global float* C,
                             int A_rows, int A_cols, int B_cols) {
        int i = get_global_id(0);
        int j = get_global_id(1);
        if (i < A_rows && j < B_cols) {
            float sum = 0.0f;
            for (int k = 0; k < A_cols; ++k) {
                sum += A[i * A_cols + k] * B[k * B_cols + j];
            }
            C[i * B_cols + j] = sum;
        }
    }
)";

    // Tiled matrix multiplication kernel (uses local memory). Tile size chosen at compile-time.
    const std::string kernel_source_matrix_mul_tiled = R"(
    #define TS 16
    __kernel void matmul_tiled(__global const float* A,
                               __global const float* B,
                               __global float* C,
                               int A_rows, int A_cols, int B_cols) {
        int row = get_global_id(0);
        int col = get_global_id(1);

        __local float As[TS][TS];
        __local float Bs[TS][TS];

        float acc = 0.0f;
        int numTiles = (A_cols + TS - 1) / TS;
        for (int t = 0; t < numTiles; ++t) {
            int localRow = get_local_id(0);
            int localCol = get_local_id(1);

            int aRow = row;
            int aCol = t * TS + localCol;
            int bRow = t * TS + localRow;
            int bCol = col;

            As[localRow][localCol] = (aRow < A_rows && aCol < A_cols) ? A[aRow * A_cols + aCol] : 0.0f;
            Bs[localRow][localCol] = (bRow < A_cols && bCol < B_cols) ? B[bRow * B_cols + bCol] : 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);

            for (int k = 0; k < TS; ++k) {
                acc += As[localRow][k] * Bs[k][localCol];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row < A_rows && col < B_cols)
            C[row * B_cols + col] = acc;
    }
    )";

// --- KernelCache ---

void KernelCache::compileKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    if (initialized) return;

    std::cout << "Compiling OpenCL kernels..." << std::endl;
    try {
        cl::Program prog_fill = loadAndBuildProgram(context, devices, kernel_source_fill, "fill");
        kernel_fill = cl::Kernel(prog_fill, "fill");

        cl::Program prog_add = loadAndBuildProgram(context, devices, kernel_source_add, "add");
        kernel_add = cl::Kernel(prog_add, "add");

        cl::Program prog_sub_mul = loadAndBuildProgram(context, devices, kernel_source_sub_mul, "sub_mul");
        kernel_sub_mul = cl::Kernel(prog_sub_mul, "sub_mul");

        cl::Program prog_transpose = loadAndBuildProgram(context, devices, kernel_source_transpose, "transpose");
        kernel_transpose = cl::Kernel(prog_transpose, "transpose");

        cl::Program prog_matrix_mul = loadAndBuildProgram(context, devices, kernel_source_matrix_mul, "matrix_mul");
        kernel_matrix_mul = cl::Kernel(prog_matrix_mul, "matrix_mul");

            // compile tiled kernel
            cl::Program prog_matrix_mul_tiled = loadAndBuildProgram(context, devices, kernel_source_matrix_mul_tiled, "matmul_tiled");
            kernel_matrix_mul_tiled = cl::Kernel(prog_matrix_mul_tiled, "matmul_tiled");

        initialized = true;
        std::cout << "OpenCL kernels compiled successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Failed to compile one or more OpenCL kernels. Aborting." << std::endl;
        throw;
    }
}

// --- MatrixCL Static Methods ---

void MatrixCL::initializeKernels(cl::Context context, const std::vector<cl::Device>& devices) {
    try {
        if (!kernels_ || !kernels_->initialized) {
            std::cout << "Creating and compiling kernels..." << std::endl;
            kernels_ = std::make_shared<KernelCache>();
            kernels_->compileKernels(context, devices);
        }
    } catch (const cl::Error& err) {
        std::cerr << "OpenCL error in kernel initialization: "
                  << err.what() << " (" << err.err() << ")" << std::endl;
        throw;
    } catch (const std::exception& e) {
        std::cerr << "Exception in kernel initialization: " << e.what() << std::endl;
        throw;
    }
}

// --- MatrixCL Implementation ---

size_t MatrixCL::buffer_size_bytes() const {
    return static_cast<size_t>(rows_) * cols_ * sizeof(float);
}

MatrixCL::MatrixCL(int rows, int cols, cl::Context context, cl::CommandQueue queue, const std::vector<float>* initial_data)
    : rows_(rows), cols_(cols), context_(context), queue_(queue)
{
    if (rows_ < 0 || cols_ < 0)
    {
        throw std::invalid_argument("MatrixCL dimensions must be non-negative");
    }

    const size_t size = buffer_size_bytes();
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, size == 0 ? sizeof(float) : size);

    if (initial_data)
    {
        if (initial_data->size() != static_cast<size_t>(rows_ * cols_))
        {
            throw std::invalid_argument("Initial data size does not match matrix dimensions");
        }
        if (size > 0)
        {
            queue_.enqueueWriteBuffer(buffer_, CL_TRUE, 0, size, initial_data->data());
        }
    }
}

MatrixCL::MatrixCL(const MatrixCL& other)
    : rows_(other.rows_), cols_(other.cols_),
      context_(other.context_), queue_(other.queue_)
{
    const size_t size = buffer_size_bytes();
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, size == 0 ? sizeof(float) : size);
    if (size > 0)
    {
        queue_.enqueueCopyBuffer(other.buffer_, buffer_, 0, 0, size);
    }
}

MatrixCL& MatrixCL::operator=(const MatrixCL& other)
{
    if (this == &other) return *this;

    rows_ = other.rows_;
    cols_ = other.cols_;
    context_ = other.context_;
    queue_ = other.queue_;

    const size_t size = buffer_size_bytes();
    buffer_ = cl::Buffer(context_, CL_MEM_READ_WRITE, size == 0 ? sizeof(float) : size);
    if (size > 0)
    {
        queue_.enqueueCopyBuffer(other.buffer_, buffer_, 0, 0, size);
    }

    return *this;
}

int MatrixCL::numRows() const { return rows_; }
int MatrixCL::numCols() const { return cols_; }
cl::Context MatrixCL::getContext() const { return context_; }
cl::CommandQueue MatrixCL::getQueue() const { return queue_; }
const cl::Buffer& MatrixCL::getBuffer() const { return buffer_; }

std::vector<float> MatrixCL::copyToHost() const
{
    std::vector<float> host_data(static_cast<size_t>(rows_) * cols_);
    size_t size = buffer_size_bytes();
    if (size == 0) return host_data;

    queue_.enqueueReadBuffer(buffer_, CL_TRUE, 0, size, host_data.data());

    return host_data;
}

void MatrixCL::fill(float value)
{
    if (rows_ * cols_ == 0) return;

    if (!kernels_ || !kernels_->initialized)
    {
        throw std::runtime_error("OpenCL kernels are not initialized");
    }

    kernels_->kernel_fill.setArg(0, buffer_);
    kernels_->kernel_fill.setArg(1, value);
    kernels_->kernel_fill.setArg(2, rows_);
    kernels_->kernel_fill.setArg(3, cols_);

    queue_.enqueueNDRangeKernel(
        kernels_->kernel_fill,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(rows_) * cols_),
        cl::NullRange);
}

MatrixCL MatrixCL::operator+(const MatrixCL& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("MatrixCL dimensions must match for addition");
    }

    MatrixCL result(rows_, cols_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    if (!kernels_ || !kernels_->initialized)
    {
        throw std::runtime_error("OpenCL kernels are not initialized");
    }

    kernels_->kernel_add.setArg(0, buffer_);
    kernels_->kernel_add.setArg(1, other.buffer_);
    kernels_->kernel_add.setArg(2, result.buffer_);
    kernels_->kernel_add.setArg(3, rows_);
    kernels_->kernel_add.setArg(4, cols_);

    queue_.enqueueNDRangeKernel(
        kernels_->kernel_add,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(rows_) * cols_),
        cl::NullRange);

    return result;
}

MatrixCL MatrixCL::operator-(const MatrixCL& other) const
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("MatrixCL dimensions must match for subtraction");
    }

    MatrixCL result(*this);
    if (rows_ * cols_ == 0) return result;

    result.sub_mul(1.0f, other);

    return result;
}

MatrixCL MatrixCL::operator*(float scalar) const
{
    MatrixCL result(rows_, cols_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    result.fill(0.0f);
    result.sub_mul(-scalar, *this);

    return result;
}

MatrixCL MatrixCL::operator*(const MatrixCL& other) const
{
    if (this->cols_ != other.rows_)
    {
        throw std::invalid_argument("MatrixCL dimensions must match for multiplication");
    }

    int C_rows = this->rows_;
    int C_cols = other.cols_;
    MatrixCL result(C_rows, C_cols, context_, queue_);
    if (C_rows * C_cols == 0) return result;

    if (!kernels_ || !kernels_->initialized)
    {
        throw std::runtime_error("OpenCL kernels are not initialized");
    }

    kernels_->kernel_matrix_mul.setArg(0, this->buffer_);
    kernels_->kernel_matrix_mul.setArg(1, other.buffer_);
    kernels_->kernel_matrix_mul.setArg(2, result.buffer_);
    kernels_->kernel_matrix_mul.setArg(3, this->rows_);
    kernels_->kernel_matrix_mul.setArg(4, this->cols_);
    kernels_->kernel_matrix_mul.setArg(5, other.cols_);

    cl::Event event;
    queue_.enqueueNDRangeKernel(
        kernels_->kernel_matrix_mul,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(this->rows_), static_cast<size_t>(other.cols_)),
        cl::NullRange,
        nullptr,
        &event);
    event.wait();

    // profiling: report kernel time in milliseconds
    try {
        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double ms = static_cast<double>(end - start) * 1e-6;
        std::cout << "[MatrixCL::operator*] kernel time (ms): " << ms << std::endl;
    } catch (...) {
        // profiling info may not be available
    }

    return result;
}

MatrixCL MatrixCL::transpose() const
{
    MatrixCL result(cols_, rows_, context_, queue_);
    if (rows_ * cols_ == 0) return result;

    if (!kernels_ || !kernels_->initialized)
    {
        throw std::runtime_error("OpenCL kernels are not initialized");
    }

    kernels_->kernel_transpose.setArg(0, buffer_);
    kernels_->kernel_transpose.setArg(1, result.buffer_);
    kernels_->kernel_transpose.setArg(2, rows_);
    kernels_->kernel_transpose.setArg(3, cols_);

    queue_.enqueueNDRangeKernel(
        kernels_->kernel_transpose,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(rows_), static_cast<size_t>(cols_)),
        cl::NullRange);

    return result;
}

MatrixCL MatrixCL::multiplyTiled(const MatrixCL& other, int tileSize) const
{
    if (this->cols_ != other.rows_)
    {
        throw std::invalid_argument("MatrixCL dimensions must match for multiplication");
    }

    int C_rows = this->rows_;
    int C_cols = other.cols_;
    MatrixCL result(C_rows, C_cols, context_, queue_);
    if (C_rows * C_cols == 0) return result;

    if (!kernels_ || !kernels_->initialized)
    {
        throw std::runtime_error("OpenCL kernels are not initialized");
    }

    // set args
    kernels_->kernel_matrix_mul_tiled.setArg(0, this->buffer_);
    kernels_->kernel_matrix_mul_tiled.setArg(1, other.buffer_);
    kernels_->kernel_matrix_mul_tiled.setArg(2, result.buffer_);
    kernels_->kernel_matrix_mul_tiled.setArg(3, this->rows_);
    kernels_->kernel_matrix_mul_tiled.setArg(4, this->cols_);
    kernels_->kernel_matrix_mul_tiled.setArg(5, other.cols_);

    // choose local size equal to tile size (TS in kernel is fixed at 16); ensure multiples
    const size_t TS = 16;
    size_t globalRow = ((size_t)C_rows + TS - 1) / TS * TS;
    size_t globalCol = ((size_t)C_cols + TS - 1) / TS * TS;

    cl::Event event;
    queue_.enqueueNDRangeKernel(
        kernels_->kernel_matrix_mul_tiled,
        cl::NullRange,
        cl::NDRange(globalRow, globalCol),
        cl::NDRange(TS, TS),
        nullptr,
        &event);
    event.wait();

    try {
        cl_ulong start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double ms = static_cast<double>(end - start) * 1e-6;
        std::cout << "[MatrixCL::multiplyTiled] kernel time (ms): " << ms << std::endl;
    } catch (...) {
    }

    return result;
}

void MatrixCL::sub_mul(float scalar, const MatrixCL& other)
{
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
        throw std::invalid_argument("MatrixCL dimensions must match for sub_mul");
    }

    if (rows_ * cols_ == 0) return;

    if (!kernels_ || !kernels_->initialized)
    {
        throw std::runtime_error("OpenCL kernels are not initialized");
    }

    kernels_->kernel_sub_mul.setArg(0, buffer_);
    kernels_->kernel_sub_mul.setArg(1, other.buffer_);
    kernels_->kernel_sub_mul.setArg(2, scalar);
    kernels_->kernel_sub_mul.setArg(3, rows_);
    kernels_->kernel_sub_mul.setArg(4, cols_);

    queue_.enqueueNDRangeKernel(
        kernels_->kernel_sub_mul,
        cl::NullRange,
        cl::NDRange(static_cast<size_t>(rows_) * cols_),
        cl::NullRange);
}
