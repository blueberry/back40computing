/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a scan of the License at
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
 * Simple program for evaluating grid size
 ******************************************************************************/

#include <stdio.h> 
#include <algorithm>

// Sorting includes
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/scan/problem_type.cuh>
#include <b40c/scan/policy.cuh>
#include <b40c/scan/enactor.cuh>

// Test utils
#include "b40c_test_util.h"


/******************************************************************************
 * Problem / Tuning Policy Types
 ******************************************************************************/

typedef int T;
typedef int SizeT;

/**
 * Sum binary scan operator
 */
template <typename T>
struct Sum
{
	// Associative reduction operator
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}

	// Identity operator
	__host__ __device__ __forceinline__ T operator()()
	{
		return 0;
	}
};

typedef b40c::scan::ProblemType <
	T,
	SizeT,
	Sum<T>,					// Reduction
	Sum<T>,					// Identity
	true,					// EXCLUSIVE
	true>					// COMMUTATIVE
		ProblemType;


typedef b40c::scan::Policy<
	ProblemType,
	200,
	b40c::util::io::ld::NONE, 		// READ_MODIFIER
	b40c::util::io::st::NONE, 		// WRITE_MODIFIER
	false, 						// UNIFORM_SMEM_ALLOCATION
	false, 						// UNIFORM_GRID_SIZE
	false, 						// OVERSUBSCRIBED_GRID_SIZE
	10,  						// LOG_SCHEDULE_GRANULARITY
	8, 7, 2, 0, 5,
	5, 2, 0, 5,
	8, 7, 1, 1, 5>
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
	b40c::scan::Enactor enactor;
	enactor.ENACTOR_DEBUG = verbose;

	b40c::GpuTimer timer;
	Sum<T> scan_op;

	printf("Sample, Items, CTAs, Elapsed, Throughput\n");
	for (int i = 0; i < samples; i++) {

		timer.Start();
		enactor.Scan<Policy>(
			d_out,
			d_in,
			num_elements,
			scan_op,
			scan_op,
			cta_sizes[i]);
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

