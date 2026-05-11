#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cassert>
#include "../include/matrix_opencl.hpp"

cl::Context context;
cl::CommandQueue queue;

static void setupOpenCL()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    assert(!platforms.empty());

    cl::Platform platform = platforms.front();
    std::cout << "Platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }
    assert(!devices.empty());

    cl::Device device = devices.front();
    std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    context = cl::Context(device);
    queue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);

    MatrixCL::initializeKernels(context, {device});
}

int main()
{
    setupOpenCL();

    std::vector<int> sizes = {64, 128, 256, 512};

    std::cout << "Matrix,NaiveKernelTime_ms,TiledKernelTime_ms,Speedup" << std::endl;

    for (int size : sizes) {
        try {
            std::vector<float> host_data_a(static_cast<size_t>(size) * size, 1.0f);
            std::vector<float> host_data_b(static_cast<size_t>(size) * size, 1.0f);

            MatrixCL A(size, size, context, queue, &host_data_a);
            MatrixCL B(size, size, context, queue, &host_data_b);

            A.fill(1.0f);
            B.fill(1.0f);

            MatrixCL warmup = A * B;
            (void)warmup;

            auto start_naive = std::chrono::high_resolution_clock::now();
            MatrixCL naive_result = A * B;
            auto end_naive = std::chrono::high_resolution_clock::now();
            double naive_time_ms = std::chrono::duration<double, std::milli>(end_naive - start_naive).count();

            double tiled_time_ms = -1.0;
            try {
                auto start_tiled = std::chrono::high_resolution_clock::now();
                MatrixCL tiled_result = A.multiplyTiled(B);
                auto end_tiled = std::chrono::high_resolution_clock::now();
                tiled_time_ms = std::chrono::duration<double, std::milli>(end_tiled - start_tiled).count();
                (void)tiled_result;
            } catch (const std::exception& e) {
                std::cerr << "Tiled multiply unavailable for size " << size << ": " << e.what() << std::endl;
            }

            double speedup = (tiled_time_ms > 0.0) ? naive_time_ms / tiled_time_ms : 1.0;

            std::cout << size << "," << std::fixed << std::setprecision(6)
                      << naive_time_ms << "," << tiled_time_ms << "," << speedup << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error with size " << size << ": " << e.what() << std::endl;
        }
    }

    std::cout << "\nBenchmark complete." << std::endl;
    return 0;
}
