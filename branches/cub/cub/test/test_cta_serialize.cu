/******************************************************************************
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
 *
 * CTA-serialization benchmark.  Each CTA waits to proceed until the previous
 * completes.
 *
 * Duane Merrill
 *
 *****************************************************************************/

#include <stdio.h>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

/******************************************************************************
 * Debug
 ******************************************************************************/

cudaError_t Debug(cudaError_t error, const char *funcname, const char *filename, int line)
{
    if (error) {
        fprintf(stderr, "[%s:%d %s] (CUDA error %d: %s)\n", filename, line, funcname, error, cudaGetErrorString(error));
        fflush(stderr);
    }
    return error;
}

/**
 * Debug macro
 */
#define CubDebug(f) Debug(f, #f, __FILE__, __LINE__)


/**
 * Debug macro with exit
 */
#define CubDebugExit(f) if (Debug(f, #f, __FILE__, __LINE__)) exit(1)


/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
class CommandLineArgs
{
protected:

    std::map<std::string, std::string> pairs;

public:

    /**
     * Constructor
     */
    CommandLineArgs(int argc, char **argv)
    {
        using namespace std;

        for (int i = 1; i < argc; i++)
        {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-')) {
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find( '=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }
            pairs[key] = val;
        }
    }


    /**
     * Checks whether a flag "--<flag>" is present in the commandline
     */
    bool CheckCmdLineFlag(const char* arg_name)
    {
        using namespace std;
        map<string, string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end()) {
            return true;
        }
        return false;
    }


    /**
     * Returns the value specified for a given commandline parameter --<flag>=<value>
     */
    template <typename T>
    void GetCmdLineArgument(const char *arg_name, T &val)
    {
        using namespace std;
        map<string, string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end()) {
            istringstream str_stream(itr->second);
            str_stream >> val;
        }
    }


    /**
     * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
     */
    template <typename T>
    void GetCmdLineArguments(const char *arg_name, std::vector<T> &vals)
    {
        using namespace std;

        // Recover multi-value string
        map<string, string>::iterator itr;
        if ((itr = pairs.find(arg_name)) != pairs.end()) {

            // Clear any default values
            vals.clear();

            string val_string = itr->second;
            istringstream str_stream(val_string);
            string::size_type old_pos = 0;
            string::size_type new_pos = 0;

            // Iterate comma-separated values
            T val;
            while ((new_pos = val_string.find(',', old_pos)) != string::npos) {

                if (new_pos != old_pos) {
                    str_stream.width(new_pos - old_pos);
                    str_stream >> val;
                    vals.push_back(val);
                }

                // skip over comma
                str_stream.ignore(1);
                old_pos = new_pos + 1;
            }

            // Read last value
            str_stream >> val;
            vals.push_back(val);
        }
    }


    /**
     * The number of pairs parsed
     */
    int ParsedArgc()
    {
        return pairs.size();
    }

    /**
     * Initialize device
     */
    cudaError_t DeviceInit(int dev = -1)
    {
        cudaError_t error = cudaSuccess;

        do {
            int deviceCount;
            error = CubDebug(cudaGetDeviceCount(&deviceCount));
            if (error) break;

            if (deviceCount == 0) {
                fprintf(stderr, "No devices supporting CUDA.\n");
                exit(1);
            }
            if (dev < 0)
            {
                GetCmdLineArgument("device", dev);
            }
            if ((dev > deviceCount - 1) || (dev < 0))
            {
                dev = 0;
            }

            cudaDeviceProp deviceProp;
            error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
            if (error) break;

            if (deviceProp.major < 1) {
                fprintf(stderr, "Device does not support CUDA.\n");
                exit(1);
            }
            if (!CheckCmdLineFlag("quiet")) {
                printf("Using device %d: %s\n", dev, deviceProp.name);
                fflush(stdout);
            }

            error = CubDebug(cudaSetDevice(dev));
            if (error) break;

        } while (0);

        return error;
    }
};


/******************************************************************************
 * Timing
 ******************************************************************************/

/**
 * Flag-based kernel performance timer
 */
struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float ElapsedMillis()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


/******************************************************************************
 * GPU device routines
 ******************************************************************************/

/**
 * Load global
 */
__device__ __forceinline__ int LoadCg(int *ptr)
{
    int val;

#if defined(_WIN64) || defined(__LP64__)
    asm volatile ("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "l"(ptr));
#else
    asm volatile ("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "r"(ptr));
#endif

    return val;
}

/**
 * Cta-serialization kernel.  Each CTA waits to proceed until the previous completes.
 */
__global__ void Kernel(
    int *d_progress,    ///< Counter indicating which CTA is allowed to complete next
    int *d_block_id)    ///< Counter for obtaining a "resident" block ID
{
    __shared__ int sblock_id;

    // Get a unique block ID that guarantees the previous block
    // has already activated.
    if (threadIdx.x == 0) sblock_id = atomicAdd(d_block_id, 1);

    __syncthreads();

    int block_id = sblock_id;
    if (threadIdx.x == 0)
    {
        if (block_id != 0)
        {
            // Wait for previous block to complete
            while (true)
            {
                if (LoadCg(d_progress) == block_id) break;
                if (LoadCg(d_progress) == block_id) break;
            }
        }

        // Signal the next CTA
        *d_progress = block_id + 1;
    }
}


/**
 * Cta-serialization kernel.  Each CTA waits to proceed until the previous completes.
 * Prints clocks counts
 */
__global__ void Kernel2(
    int *d_progress,    ///< Counter indicating which CTA is allowed to complete next
    int *d_block_id,    ///< Counter for obtaining a "resident" block ID
    int *d_clocks,
    int *d_wait_cycles)
{
    if (threadIdx.x == 0)
    {
        int iterations = 0;
        int a = clock();
        if (blockIdx.x > 0)
        {
            // Wait for previous block to complete
            while (true)
            {
                if (LoadCg(d_progress) == blockIdx.x) break;
                iterations++;
                if (LoadCg(d_progress) == blockIdx.x) break;
                iterations++;
            }
        }
        d_progress[0] = blockIdx.x + 1;
        int b = clock();

        d_clocks[blockIdx.x] = b - a;
        d_wait_cycles[blockIdx.x] = iterations;
    }
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    int iterations      = 100;
    int num_ctas        = 1024 * 63;
    int cta_size        = 32;
    int occupancy       = -1;

    CommandLineArgs args(argc, argv);
    CubDebugExit(args.DeviceInit());
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("num-ctas", num_ctas);
    args.GetCmdLineArgument("cta-size", cta_size);
    args.GetCmdLineArgument("occupancy", occupancy);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--i=<iterations>] "
            "[--cta-size=<cta-size>] "
            "[--num-ctas=<num-ctas>] "
            "[--occupancy=<occupancy>] "
            "\n", argv[0]);
        exit(0);
    }

    int dynamic_smem = (occupancy <= 0) ?
        0 :
        (1024 * 48 - 128) / occupancy;

    printf("%d iterations of Kernel<<<%d, %d, %d>>>(...)\n", iterations, num_ctas, cta_size, dynamic_smem);
    fflush(stdout);

    // Device storage
    int *d_progress, *d_block_id, *d_clocks, *d_wait_cycles;

    // Allocate device words
    CubDebugExit(cudaMalloc((void**)&d_progress, sizeof(int)));
    CubDebugExit(cudaMalloc((void**)&d_block_id, sizeof(int)));

    /**
     * Experiment 1: CTA-serialization throughput.
     */

    printf("Experiment 1: CTA-serialization throughput.\n");
    fflush(stdout);


    GpuTimer gpu_timer;
    float elapsed_millis = 0.0;
    for (int i = 0; i < iterations; i++)
    {
        // Zero-out the counters
        CubDebugExit(cudaMemset(d_progress, 0, sizeof(int)));
        CubDebugExit(cudaMemset(d_block_id, 0, sizeof(int)));

        gpu_timer.Start();

        Kernel<<<num_ctas, cta_size, dynamic_smem>>>(d_progress, d_block_id);

        gpu_timer.Stop();
        elapsed_millis += gpu_timer.ElapsedMillis();
    }
    float avg_elapsed = elapsed_millis / iterations;

    printf("%d iterations, average elapsed (%.4f ms), %.4f M CTAs/s\n",
        iterations,
        avg_elapsed,
        float(num_ctas) / avg_elapsed / 1000.0);
    fflush(stdout);



    /**
     * Experiment 2: Launch 1 "sequentialized" CTA per SM, record the number of clocks
     * elapsed until each is able to reture.
     */

    printf("\n\nBenchmark 2: clocks per retired CTA\n");
    fflush(stdout);

    num_ctas = 7;
    cta_size = 32;
    CubDebugExit(cudaMalloc((void**)&d_clocks, sizeof(int) * num_ctas));
    CubDebugExit(cudaMalloc((void**)&d_wait_cycles, sizeof(int) * num_ctas));
    CubDebugExit(cudaMemset(d_clocks, 0, sizeof(int) * num_ctas));
    CubDebugExit(cudaMemset(d_wait_cycles, 0, sizeof(int) * num_ctas));

    CubDebugExit(cudaMemset(d_progress, 0, sizeof(int)));
    CubDebugExit(cudaMemset(d_block_id, 0, sizeof(int)));
    Kernel2<<<num_ctas, cta_size, 1024 * 40>>>(d_progress, d_block_id, d_clocks, d_wait_cycles);

    int *h_clocks = new int[num_ctas];
    int *h_wait_cycles = new int[num_ctas];

    CubDebug(cudaMemcpy(h_clocks, d_clocks, sizeof(int) * num_ctas, cudaMemcpyDeviceToHost));
    CubDebug(cudaMemcpy(h_wait_cycles, d_wait_cycles, sizeof(int) * num_ctas, cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_ctas; i++)
    {
        printf("Cta %d clocks(%d) wait_cycles(%d), avg clocks per predecessor(%.2f)\n",
            i, h_clocks[i], h_wait_cycles[i], (i == 0) ? 0 : float(h_clocks[i]) / i);
    }

    // Force any kernel stdio to screen
    CubDebugExit(cudaThreadSynchronize());

    // Cleanup
    CubDebugExit(cudaFree(d_progress));
    CubDebugExit(cudaFree(d_block_id));
    CubDebugExit(cudaFree(d_clocks));
    CubDebugExit(cudaFree(d_wait_cycles));

    delete[] h_clocks;
    delete[] h_wait_cycles;

    return 0;
}
