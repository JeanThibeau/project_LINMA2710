#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "matrix.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
Matrix makeMatrix(int n)
{
    Matrix m(n, n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            // Deterministic values to avoid RNG overhead in benchmark setup.
            m.set(i, j, ((i * 17 + j * 31) % 100) / 100.0);
        }
    }
    return m;
}

double benchmarkMultiply(const Matrix &a, const Matrix &b, int repeats)
{
    using clock = std::chrono::high_resolution_clock;
    double totalMs = 0.0;

    for (int r = 0; r < repeats; ++r)
    {
        const auto t0 = clock::now();
        volatile Matrix c = a * b;
        const auto t1 = clock::now();
        (void)c;

        totalMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    return totalMs / repeats;
}
} // namespace

int main()
{
    const std::vector<int> sizes = {128, 256, 384, 512};
    const int repeats = 3;

#ifdef _OPENMP
    const int maxThreads = omp_get_max_threads();
#else
    const int maxThreads = 1;
#endif

    std::vector<int> threadCounts = {1, 2, 4, 8};
    threadCounts.erase(
        std::remove_if(
            threadCounts.begin(),
            threadCounts.end(),
            [maxThreads](int t)
            {
                return t > maxThreads;
            }),
        threadCounts.end());
    if (threadCounts.empty())
    {
        threadCounts.push_back(1);
    }

    std::cout << "Matrix multiplication benchmark (average over " << repeats << " runs)\n";
    std::cout << "Max available OpenMP threads: " << maxThreads << "\n\n";
    std::cout << std::left << std::setw(12) << "size"
              << std::setw(12) << "threads"
              << std::setw(16) << "time_ms"
              << std::setw(14) << "speedup"
              << "gflops" << "\n";

    for (int n : sizes)
    {
        Matrix a = makeMatrix(n);
        Matrix b = makeMatrix(n);

        double baselineMs = 0.0;

        for (int threads : threadCounts)
        {
#ifdef _OPENMP
            omp_set_num_threads(threads);
#endif

            const double ms = benchmarkMultiply(a, b, repeats);
            if (threads == 1)
            {
                baselineMs = ms;
            }

            const double speedup = baselineMs / ms;
            const double flops = 2.0 * static_cast<double>(n) * n * n;
            const double gflops = flops / (ms * 1e6);

            std::cout << std::left << std::setw(12) << n
                      << std::setw(12) << threads
                      << std::setw(16) << std::fixed << std::setprecision(3) << ms
                      << std::setw(14) << std::setprecision(2) << speedup
                      << std::setprecision(2) << gflops << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
