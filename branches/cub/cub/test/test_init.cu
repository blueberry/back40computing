// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <iterator>
#include <vector>
#include <algorithm>
#include <stdio.h>
#include <test_util.h>

#include "../cub.cuh"

using namespace cub;
using namespace std;

bool    g_verbose       = false;
int     g_iterations    = 100000;
int     g_num_counters  = 4;


__global__ void MemsetKernel(int *d_counters)
{
    if (threadIdx.x == 0)
    {
        d_counters[0] = 0;
        d_counters[1] = 0;
        d_counters[2] = 0;
        d_counters[3] = 0;
    }
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("i", g_iterations);

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Device storage
    int *d_counters;

    // Allocate device words
    CachedAllocator *allocator = CubCachedAllocator<void>();
    CubDebugExit(allocator->Allocate((void**)&d_counters, sizeof(int) * g_num_counters));

    GpuTimer gpu_timer;

    //
    // Run cudaMemset
    //

    gpu_timer.Start();
    for (int i = 0; i < g_iterations; i++)
    {
        // Zero-out the counters
        CubDebugExit(cudaMemset(d_counters + 0, 0, sizeof(int)));
        CubDebugExit(cudaMemset(d_counters + 1, 0, sizeof(int)));
        CubDebugExit(cudaMemset(d_counters + 2, 0, sizeof(int)));
        CubDebugExit(cudaMemset(d_counters + 3, 0, sizeof(int)));
    }
    gpu_timer.Stop();
    printf("cudaMemset %d iterations, average elapsed (%.4f ms)\n",
        g_iterations,
        gpu_timer.ElapsedMillis() / g_iterations);

    //
    // Run cudaMemsetAsync
    //

    gpu_timer.Start();
    for (int i = 0; i < g_iterations; i++)
    {
        // Zero-out the counters
        CubDebugExit(cudaMemsetAsync(d_counters + 0, 0, sizeof(int)));
        CubDebugExit(cudaMemsetAsync(d_counters + 1, 0, sizeof(int)));
        CubDebugExit(cudaMemsetAsync(d_counters + 2, 0, sizeof(int)));
        CubDebugExit(cudaMemsetAsync(d_counters + 3, 0, sizeof(int)));
    }
    gpu_timer.Stop();
    printf("cudaMemsetAsync %d iterations, average elapsed (%.4f ms)\n",
        g_iterations,
        gpu_timer.ElapsedMillis() / g_iterations);

    //
    // Run MemsetKernel
    //

    gpu_timer.Start();
    for (int i = 0; i < g_iterations; i++)
    {
        // Zero-out the counters
        MemsetKernel<<<1, g_num_counters>>>(d_counters);
    }
    gpu_timer.Stop();
    printf("MemsetKernel %d iterations, average elapsed (%.4f ms)\n",
        g_iterations,
        gpu_timer.ElapsedMillis() / g_iterations);


    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());

    // Cleanup
    CubDebugExit(allocator->DeviceFree(d_counters));

    return 0;
}
