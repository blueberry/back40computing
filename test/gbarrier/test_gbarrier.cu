/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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
    int occupancy = B40C_MIN(
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
