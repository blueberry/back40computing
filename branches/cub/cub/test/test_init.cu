#include <stdio.h>

#include <test_util.h>

__device__ __forceinline__ int LoadCg(int *ptr)
{
    int val;
    asm volatile ("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "r"(ptr));
    return val;
}


__global__ void KernelA(int *d_counter, int *d_bid)
{
    __shared__ int sbid;

    if (threadIdx.x == 0)
    {
        sbid = atomicAdd(d_bid, 1);
    }

    __syncthreads();

    int bid = sbid;
    if (threadIdx.x == 0)
    {
        // wait for prev
        if (bid != 0)
        {
            while (LoadCg(d_counter + bid) == 0)
            {
                __threadfence_block();
            }
        }

        // set next
        d_counter[bid + 1] = 1;
    }
}



/**
 * Main
 */
int main(int argc, char** argv)
{
    int iterations      = 100;
    int num_ctas        = 1024 * 63;
    int cta_size        = 1024;

    CommandLineArgs args(argc, argv);
    CubDebugExit(args.DeviceInit());
    args.GetCmdLineArgument("i", iterations);

    // Device storage
    int *d_counter, *d_bid;

    // Allocate device words
    CubDebugExit(cudaMalloc((void**)&d_counter, sizeof(int) * num_ctas));
    CubDebugExit(cudaMalloc((void**)&d_bid, sizeof(int)));

    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < iterations; i++)
    {
        // Zero-out the counters
        CubDebugExit(cudaMemset(d_counter, 0, sizeof(int) * num_ctas));
        CubDebugExit(cudaMemset(d_bid, 0, sizeof(int)));

        gpu_timer.Start();

        KernelA<<<num_ctas, cta_size>>>(d_counter, d_bid);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }

    float avg_elapsed = elapsed_millis / iterations;

    printf("%d iterations, average elapsed (%.4f ms), %.4f M CTAs/s\n",
        iterations,
        avg_elapsed,
        float(num_ctas) / avg_elapsed / 1000.0);

    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());

    // Cleanup
    CubDebugExit(cudaFree(d_counter));
    CubDebugExit(cudaFree(d_bid));

    return 0;
}
