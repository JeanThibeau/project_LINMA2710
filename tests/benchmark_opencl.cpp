#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "../include/matrix_opencl.hpp"

int main() {
    std::vector<int> sizes = {64, 128, 256, 512};
    
    std::cout << "Matrix,NaiveKernelTime_ms,TiledKernelTime_ms,Speedup" << std::endl;
    
    for (int size : sizes) {
        try {
            MatrixCL A(size, size);
            MatrixCL B(size, size);
            MatrixCL C(size, size);
            
            A.fill(1.0f);
            B.fill(1.0f);
            
            // Warm up
            MatrixCL _ = A * B;
            
            // Measure naive kernel (operator*)
            auto start_naive = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < 5; ++i) {
                C = A * B;
            }
            auto end_naive = std::chrono::high_resolution_clock::now();
            double naive_time_ms = std::chrono::duration<double, std::milli>(end_naive - start_naive).count() / 5.0;
            
            // Measure tiled kernel (if implemented)
            double tiled_time_ms = naive_time_ms; // fallback
            try {
                auto start_tiled = std::chrono::high_resolution_clock::now();
                for (int i = 0; i < 5; ++i) {
                    C = A.multiplyTiled(B);
                }
                auto end_tiled = std::chrono::high_resolution_clock::now();
                tiled_time_ms = std::chrono::duration<double, std::milli>(end_tiled - start_tiled).count() / 5.0;
            } catch (...) {
                // multiplyTiled not available, skip
                tiled_time_ms = -1.0;
            }
            
            double speedup = (tiled_time_ms > 0) ? naive_time_ms / tiled_time_ms : 1.0;
            
            std::cout << size << "," << std::fixed << std::setprecision(6) 
                      << naive_time_ms << "," << tiled_time_ms << "," << speedup << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error with size " << size << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "\nBenchmark complete." << std::endl;
    return 0;
}
