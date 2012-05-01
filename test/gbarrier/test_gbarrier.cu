/******************************************************************************
 *
 * Copyright 2010-2012 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For more information, see our Google Code project site:
 * http://code.google.com/p/back40computing/
 *
 ******************************************************************************/

/******************************************************************************
 * Test evaluation for software global barrier throughput
 ******************************************************************************/


#include <stdio.h>

#include "b40c_test_util.h"
#include <b40c/util/global_barrier.cuh>


/**
 * Kernel that iterates through the specified number of software global barriers
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
 * Main
 */
int main(int argc, char** argv)
{
	// Initialize device
    b40c::CommandLineArgs args(argc, argv);
    b40c::DeviceInit(args);

    // Defaults
    int iterations = 10000;
    int cta_size = 128;
    int grid_size = -1;

    // Get args
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("grid-size", grid_size);
    args.GetCmdLineArgument("cta-size", cta_size);

    // Print usage
    if (args.CheckCmdLineFlag("help")) {
    	printf("--device=<device-id> --i=<iterations> --grid-size<grid-size> --cta-size<cta-size>\n");
    	exit(0);
    }

    // Initialize CUDA device properties
    b40c::util::CudaProperties cuda_props;

    // Compute grid size and occupancy
    int occupancy = CUB_MIN(
    	(B40C_SM_THREADS(cuda_props.kernel_ptx_version) / cta_size),
    	B40C_SM_CTAS(cuda_props.kernel_ptx_version));

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

	// Output timing results
	float avg_elapsed = gpu_timer.ElapsedMillis() / float(iterations);
	printf("%d iterations, %f total elapsed millis, %f avg elapsed millis\n",
		iterations,
		gpu_timer.ElapsedMillis(),
		avg_elapsed);

	cudaError_t retval;
	retval = b40c::util::B40CPerror(cudaThreadSynchronize(), "EnactorContractExpandGBarrier Kernel failed ", __FILE__, __LINE__);

	return retval;
}
