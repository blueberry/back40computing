/******************************************************************************
 * 
 * Scanright 2010-2011 Duane Merrill
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
 * Simple test driver program for scan.
 ******************************************************************************/

#include <stdio.h> 

#include <b40c/segmented_scan/problem_type.cuh>
#include <b40c/segmented_scan/problem_config.cuh>
#include <b40c/segmented_scan/kernel_config.cuh>
#include <b40c/segmented_scan/problem_config_tuned.cuh>

#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>

#include <b40c/segmented_scan/enactor.cuh>

// Test utils
#include "b40c_test_util.h"



using namespace b40c;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/


/**
 * Sum binary associative operator
 */
template <typename T>
__host__ __device__ __forceinline__ T Sum(const T &a, const T &b)
{
	return a + b;
}


/**
 * Identity for Sum operator for integer types
 */
template <typename T>
__host__ __device__ __forceinline__ T SumId()
{
	return 0;
}



/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_copy [--device=<device index>]\n");
    	return 0;
    }

    DeviceInit(args);

    // Define our problem type
	typedef unsigned int T;
	typedef unsigned char Flag;

	const int NUM_ELEMENTS = 512;
	const bool EXCLUSIVE_SCAN = false;

	// Allocate and initialize host problem data and host reference solution
	T h_src[NUM_ELEMENTS];
	Flag h_flags[NUM_ELEMENTS];
	T h_reference[NUM_ELEMENTS];

	for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
		h_src[i] = 1;
		h_flags[i] = (i % 11) == 0;
	}

	printf("Flags:\n");
	for (int i = 0; i < NUM_ELEMENTS; i++) {
		PrintValue(h_flags[i]);
		printf(", ");
	}
	printf("\n\n");

	for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
		if (EXCLUSIVE_SCAN)
		{
			h_reference[i] = ((i == 0) || (h_flags[i])) ?
				SumId<T>() :
				Sum(h_reference[i - 1], h_src[i - 1]);
		} else {
			h_reference[i] = ((i == 0) || (h_flags[i])) ?
				h_src[i] :
				Sum(h_reference[i - 1], h_src[i]);
		}
	}

	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	Flag *d_flags;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_flags, sizeof(Flag) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);

	cudaMemcpy(d_src, h_src, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	cudaMemcpy(d_flags, h_flags, sizeof(Flag) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	
	// Exclusive segmented scan

	typedef b40c::segmented_scan::ProblemType<
		T, Flag, size_t, EXCLUSIVE_SCAN, Sum, SumId> ProblemType;

	typedef b40c::segmented_scan::ProblemConfig<
		ProblemType,
		b40c::segmented_scan::SM20,
		b40c::util::ld::NONE,
		b40c::util::st::NONE,
		false,
		false,
		false,
		7, 8, 5, 1, 1, 5,
		      5, 1, 1, 5,
		   8, 5, 1, 1, 5> CustomConfig;

	segmented_scan::Enactor segmented_scan_enactor;
	segmented_scan_enactor.DEBUG = true;
	segmented_scan_enactor.Enact<CustomConfig>(
		d_dest, d_src, d_flags, NUM_ELEMENTS);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS, true, true);
	printf("\n");


	return 0;
}

