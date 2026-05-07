#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#else
#error "OpenMP is required for this benchmark"
#endif

// Utility function to measure time (in milliseconds)
double measureTime(std::function<void()> func)
{
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "OpenMP Matrix Multiplication Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    // Test parameters
    std::vector<int> sizes = {32, 64, 128, 256, 512, 1024};
    std::vector<int> threads = {1, 2, 4, 8};

    // Get max available threads
    int max_threads = omp_get_max_threads();
    std::cout << "Max available threads: " << max_threads << std::endl;
    std::cout << std::endl;

    // Adjust thread counts based on availability
    std::vector<int> actual_threads;
    for (int t : threads)
    {
        if (t <= max_threads)
        {
            actual_threads.push_back(t);
        }
    }

    std::cout << "Matrix Sizes: ";
    for (int s : sizes)
        std::cout << s << " ";
    std::cout << std::endl;

    std::cout << "Thread Counts: ";
    for (int t : actual_threads)
        std::cout << t << " ";
    std::cout << std::endl;
    std::cout << std::endl;

    // Benchmark: vary matrix size and thread count
    std::cout << std::setw(10) << "Size";
    for (int t : actual_threads)
    {
        std::cout << std::setw(12) << ("T=" + std::to_string(t) + " ms");
        std::cout << std::setw(12) << ("T=" + std::to_string(t) + " spd");
    }
    std::cout << std::endl;

    std::cout << std::string(10 + 24 * actual_threads.size(), '-') << std::endl;

    // For each matrix size
    for (int size : sizes)
    {
        std::vector<double> times;
        std::vector<double> speedups;

        // For each thread count
        for (int num_threads : actual_threads)
        {
            omp_set_num_threads(num_threads);

            // Create matrices
            Matrix A(size, size);
            Matrix B(size, size);

            // Fill with random values
            A.fill(1.0);
            B.fill(1.0);

            // Warm-up run
            Matrix C = A * B;

            // Measure multiplication time
            double time_ms = measureTime([&]() {
                C = A * B;
            });

            times.push_back(time_ms);
        }

        const double baseline_time = times.front();
        for (double time_ms : times)
        {
            speedups.push_back(baseline_time / time_ms);
        }

        // Print results
        std::cout << std::setw(10) << size;
        for (std::size_t i = 0; i < times.size(); ++i)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << times[i];
            std::cout << std::setw(12) << std::fixed << std::setprecision(2) << speedups[i] << "x";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;

    // Detailed analysis for a specific size
    int analysis_size = 512;
    std::cout << "========================================" << std::endl;
    std::cout << "Detailed Analysis for " << analysis_size << "x" << analysis_size
              << " Matrices" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    std::cout << std::setw(10) << "Threads" << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Speedup" << std::setw(15) << "Efficiency (%)" << std::endl;
    std::cout << std::string(55, '-') << std::endl;

    double baseline_time = 0.0;

    for (std::size_t i = 0; i < actual_threads.size(); ++i)
    {
        int num_threads = actual_threads[i];
        omp_set_num_threads(num_threads);

        Matrix A(analysis_size, analysis_size);
        Matrix B(analysis_size, analysis_size);
        A.fill(1.0);
        B.fill(1.0);

        // Warm-up
        Matrix C = A * B;

        // Measure
        double time_ms = measureTime([&]() {
            C = A * B;
        });

        if (i == 0)
            baseline_time = time_ms;

        double speedup = baseline_time / time_ms;
        double efficiency = (speedup / num_threads) * 100.0;

        std::cout << std::setw(10) << num_threads << std::setw(15) << std::fixed 
                  << std::setprecision(2) << time_ms << std::setw(15) << speedup
                  << std::setw(15) << efficiency << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Notes:" << std::endl;
    std::cout << "- Speedup = Time(1 thread) / Time(N threads)" << std::endl;
    std::cout << "- Efficiency = Speedup / N * 100%" << std::endl;
    std::cout << "- Ideal efficiency is 100% (linear scaling)" << std::endl;
    std::cout << std::endl;

    return 0;
}
