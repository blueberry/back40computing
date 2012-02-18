
#include <stdio.h>


// Test utils
#include "b40c_test_util.h"
#include <b40c/util/global_barrier.cuh>

/**
 * Kernel
 */
__global__ void Kernel(
	b40c::util::GlobalBarrier global_barrier,
	int iterations)
{
	for (int i = 0; i < iterations; i++) {
		global_barrier.Sync();
	}
}




/**
 * Simple computation of SM occupancy given cta size and ptx version
 */
int Occupancy(int cta_size, int ptx_version)
{
	return B40C_MIN(
		(B40C_SM_THREADS(ptx_version) / cta_size),
		B40C_SM_CTAS(ptx_version));
}


/**
 * Main
 */
int main(int argc, char** argv)
{
    b40c::CommandLineArgs args(argc, argv);
    b40c::DeviceInit(args);

    b40c::util::CudaProperties cuda_props;

    // Defaults
    int iterations = 10000;
    int cta_size = 128;
    int grid_size = -1;

    // Get args
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("grid-size", grid_size);
    args.GetCmdLineArgument("cta-size", cta_size);

    if (args.CheckCmdLineFlag("help")) {
    	printf("--device=<device-id> --i=<iterations> --grid-size<grid-size> --cta-size<cta-size>\n");
    	exit(0);
    }

    // Compute grid size / occupancy
    int occupancy = Occupancy(cta_size, cuda_props.kernel_ptx_version);
    if (grid_size == -1) {
    	grid_size = occupancy * cuda_props.device_props.multiProcessorCount;
    } else {
    	occupancy = grid_size / cuda_props.device_props.multiProcessorCount;
    }

    printf("Initializing software global barrier for Kernel<<<%d,%d>>> with %d occupancy\n",
    	grid_size, cta_size, occupancy);

    // Init global barrier
    b40c::util::GlobalBarrierLifetime global_barrier;
	global_barrier.Setup(grid_size);

	// Time kernel
	b40c::GpuTimer gpu_timer;
	gpu_timer.Start();
	Kernel<<<grid_size, cta_size>>>(global_barrier, iterations);
	gpu_timer.Stop();

	float avg_elapsed = gpu_timer.ElapsedMillis() / float(iterations);
	printf("%d iterations, %f total elapsed millis, %f avg elapsed millis\n",
		iterations,
		gpu_timer.ElapsedMillis(),
		avg_elapsed);

	cudaError_t retval;
	retval = b40c::util::B40CPerror(cudaThreadSynchronize(), "EnactorContractExpandGBarrier Kernel failed ", __FILE__, __LINE__);

	return retval;
}
