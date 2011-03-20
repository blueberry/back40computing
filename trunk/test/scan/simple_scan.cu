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
#include <b40c/scan_enactor.cuh>

// Test utils
#include "b40c_test_util.h"

using namespace b40c;


/******************************************************************************
 * Utility Routines
 ******************************************************************************/


/**
 * Max binary associative operator
 */
template <typename T>
__host__ __device__ __forceinline__ T Max(const T &a, const T &b)
{
	return (a > b) ? a : b;
}


/**
 * Identity for max operator for unsigned integer types (i.e., max(a, identity) == a)
 */
template <typename T>
__host__ __device__ __forceinline__ T MaxId()
{
	return 0;
}


/**
 * Example showing syntax for invoking templated member functions from 
 * a templated function
 */
template <
	typename T,
	bool EXCLUSIVE_SCAN,
	T BinaryOp(const T&, const T&),
	T Identity()>
void TemplatedSubroutineScan(
	b40c::ScanEnactor &scan_enactor,
	T *d_dest, 
	T *d_src,
	int num_elements)
{
	scan_enactor.template Enact<T, EXCLUSIVE_SCAN, BinaryOp, Identity>(d_dest, d_src, num_elements);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	CommandLineArgs args(argc, argv);

	// Usage/help
    if (args.CheckCmdLineFlag("help") || args.CheckCmdLineFlag("h")) {
    	printf("\nsimple_scan [--device=<device index>]\n");
    	return 0;
    }

    DeviceInit(args);

    // Define our problem type
	typedef unsigned int T;
	const int NUM_ELEMENTS = 10;
	const bool EXCLUSIVE_SCAN = true;

	// Allocate and initialize host problem data and host reference solution
	T h_src[NUM_ELEMENTS];
	T h_reference[NUM_ELEMENTS];

	for (size_t i = 0; i < NUM_ELEMENTS; ++i) {
		h_src[i] = i;
		if (EXCLUSIVE_SCAN)
		{
			h_reference[i] = (i == 0) ?
				MaxId<T>() :
				Max(h_reference[i - 1], h_src[i - 1]);
		} else {
			h_reference[i] = (i == 0) ?
				h_src[i] :
				Max(h_reference[i - 1], h_src[i]);
		}
	}

	
	// Allocate and initialize device data
	T *d_src, *d_dest;
	cudaMalloc((void**) &d_src, sizeof(T) * NUM_ELEMENTS);
	cudaMalloc((void**) &d_dest, sizeof(T) * NUM_ELEMENTS);
	cudaMemcpy(d_src, h_src, sizeof(T) * NUM_ELEMENTS, cudaMemcpyHostToDevice);
	
	// Create a scan enactor
	b40c::ScanEnactor scan_enactor;
	

	//
	// Example 1: Enact simple exclusive scan using internal tuning heuristics
	//
	scan_enactor.Enact<T, EXCLUSIVE_SCAN, Max, MaxId>(
		d_dest, d_src, NUM_ELEMENTS);
	
	printf("Simple scan: "); CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");
	
	
	//
	// Example 2: Enact simple exclusive scan using "large problem" tuning configuration
	//
	scan_enactor.Enact<T, EXCLUSIVE_SCAN, Max, MaxId, b40c::scan::LARGE>(
		d_dest, d_src, NUM_ELEMENTS);

	printf("Large-tuned scan: "); CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");

	
	//
	// Example 3: Enact simple exclusive scan using "small problem" tuning configuration
	//
	scan_enactor.Enact<T, EXCLUSIVE_SCAN, Max, MaxId, b40c::scan::SMALL>(
		d_dest, d_src, NUM_ELEMENTS);
	
	printf("Small-tuned scan: "); CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 4: Enact simple exclusive scan using a templated subroutine function
	//
	TemplatedSubroutineScan<T, EXCLUSIVE_SCAN, Max, MaxId>(scan_enactor, d_dest, d_src, NUM_ELEMENTS);
	
	printf("Templated subroutine scan: "); CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");


	//
	// Example 5: Enact simple exclusive scan using custom tuning configuration (base scan enactor)
	//
	typedef b40c::scan::ProblemType<T, size_t, EXCLUSIVE_SCAN, Max, MaxId> ProblemType;
	typedef b40c::scan::ProblemConfig<
		ProblemType,
		b40c::scan::SM20,
		b40c::util::ld::CG,
		b40c::util::st::CG,
		false, 
		false,
		false,
		8,
		8, 7, 1, 0,
		8, 0, 1, 5,
		8, 7, 1, 0, 5> CustomConfig;
	
	scan_enactor.Enact<CustomConfig>(d_dest, d_src, NUM_ELEMENTS);

	printf("Custom scan: "); CompareDeviceResults(h_reference, d_dest, NUM_ELEMENTS); printf("\n");

	return 0;
}

