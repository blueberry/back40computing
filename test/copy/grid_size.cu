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
 * Simple program for evaluating grid size
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>

// Sorting includes
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/copy/policy.cuh>
#include <b40c/copy/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Problem / Tuning Policy Types
 ******************************************************************************/

typedef int T;
typedef int SizeT;

typedef b40c::copy::Policy <
	T,
	SizeT,
	200, 					// CUDA_ARCH,
	10,						// LOG_SCHEDULE_GRANULARITY
	1,						// MIN_CTA_OCCUPANCY
	7, 						// LOG_THREADS
	2, 						// LOG_LOAD_VEC_SIZE
	0, 						// LOG_LOADS_PER_TILE
	b40c::util::io::ld::NONE,	// READ_MODIFIER
	b40c::util::io::st::NONE, 	// WRITE_MODIFIER
	false,					// WORKSTEALING
	false> 					// OVERSUBSCRIBED
		Policy;


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
    // Initialize command line
    b40c::CommandLineArgs args(argc, argv);
    b40c::DeviceInit(args);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\ngrid_size [--device=<device index>] [--v] [--i=<samples>] [--n=<elements>]\n");
    	return 0;
    }

	// Parse commandline args
    SizeT num_elements = 1024 * 1024 * 64;			// 64 million items
    int samples = 10;								// 1 sample

    bool verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("i", samples);

    // Allocate array of random grid sizes (1 - 65536)
    int *cta_sizes = new int[samples];
	for (int i = 0; i < samples; i++) {
		b40c::util::RandomBits(cta_sizes[i], 0, 16);
		if (cta_sizes[i] == 0) cta_sizes[i] = 1;
	}

	// Allocate and initialize host problem data
	T *h_data = new T[num_elements];
	for (SizeT i = 0; i < num_elements; ++i) {
		h_data[i] = i;
	}

	// Allocate device data.
	T *d_in;
	T *d_out;
	cudaMalloc((void**) &d_in, sizeof(T) * num_elements);
	cudaMalloc((void**) &d_out, sizeof(T) * num_elements);

	cudaMemcpy(d_in, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice);

	//
	// Perform passes
	//

	// Create an enactor
	b40c::copy::Enactor enactor;
	enactor.ENACTOR_DEBUG = verbose;

	// Perform the timed number of iterations
	b40c::GpuTimer timer;

	printf("Sample, Items, CTAs, Elapsed, Throughput\n");
	for (int i = 0; i < samples; i++) {

		timer.Start();
		enactor.Copy<Policy>(d_out, d_in, num_elements, cta_sizes[i]);
		timer.Stop();

		float throughput = float(num_elements) / timer.ElapsedMillis() / 1000.0 / 1000.0;

		printf("%d, %d, %d, %f, %f\n",
			i,
			num_elements,
			cta_sizes[i],
			timer.ElapsedMillis(),
			throughput);
	}

	// Cleanup
	cudaFree(d_in);
	cudaFree(d_out);
	delete h_data;
	delete cta_sizes;

	return 0;
}

