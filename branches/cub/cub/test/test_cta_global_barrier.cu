/******************************************************************************
 *
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Test evaluation for software global barrier throughput
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>

#include <stdio.h>
#include "../cub.cuh"
#include <test_util.h>

using namespace cub;


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * Kernel that iterates through the specified number of software global barriers
 */
__global__ void Kernel(
	GridTestBarrier global_barrier,
	int iterations)
{
	for (int i = 0; i < iterations; i++)
	{
		global_barrier.Sync();
	}
}


//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
	cudaError_t retval = cudaSuccess;

	// Defaults
    int iterations = 10000;
    int cta_size = 128;
    int grid_size = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);

    // Get args
    args.GetCmdLineArgument("i", iterations);
    args.GetCmdLineArgument("grid-size", grid_size);
    args.GetCmdLineArgument("cta-size", cta_size);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
    	printf("%s "
    		"[--device=<device-id>]"
    		"[--i=<iterations>]"
    		"[--grid-size<grid-size>]"
    		"[--cta-size<cta-size>]"
    		"\n", argv[0]);
    	exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

    // Initialize CUDA device properties
    CudaProps cuda_props;
    CubDebugExit(cuda_props.Init());

    // Compute grid size and occupancy
    int occupancy = CUB_MIN(
    	(cuda_props.max_sm_threads / cta_size),
    	cuda_props.max_sm_ctas);

    if (grid_size == -1)
    {
    	grid_size = occupancy * cuda_props.sm_count;
    }
    else
    {
    	occupancy = grid_size / cuda_props.sm_count;
    }

    printf("Initializing software global barrier for Kernel<<<%d,%d>>> with %d occupancy\n",
    	grid_size, cta_size, occupancy);
    fflush(stdout);

    // Init global barrier
    GridTestBarrierLifetime global_barrier;
	global_barrier.Setup(grid_size);

	// Time kernel
	GpuTimer gpu_timer;
	gpu_timer.Start();
	Kernel<<<grid_size, cta_size>>>(global_barrier, iterations);
	gpu_timer.Stop();

	retval = CubDebug(cudaThreadSynchronize());

	// Output timing results
	float avg_elapsed = gpu_timer.ElapsedMillis() / float(iterations);
	printf("%d iterations, %f total elapsed millis, %f avg elapsed millis\n",
		iterations,
		gpu_timer.ElapsedMillis(),
		avg_elapsed);

	return retval;
}
