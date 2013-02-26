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
 * Simple test driver program for *large-problem* radix sorting.
 *
 * Useful for demonstrating how to integrate LsbEarlyExit radix sorting into 
 * your application 
 ******************************************************************************/

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>
#include <algorithm>

#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>

#include <b40c/radix_sort/enactor.cuh>

#include "b40c_test_util.h"

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

// #define ENACTOR_DEBUG
bool 	g_verbose;
bool 	g_keys_only;
int 	g_max_ctas 			= 0;
int 	g_iterations  		= 1;


/******************************************************************************
 * Test structures
 ******************************************************************************/

// Test value-type structure 
struct Fribbitz {
	char a;
	double b;
	unsigned short c;
};


/******************************************************************************
 * Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_large_problem_sorting [--device=<device index>] [--v] [--i=<num-g_iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--keys-only]\n");
	printf("\n");
	printf("\t--v\tDisplays sorted results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sorting operation <num-g_iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
	printf("\t--keys-only\tSpecifies that keys are not accommodated by value pairings\n");
	printf("\n");
}



/**
 * Uses the GPU to sort the specified vector of elements for the given
 * number of g_iterations, displaying runtime information.
 */
template <
	radix_sort::ProbSizeGenre GENRE,
	typename DoubleBuffer,
	typename SizeT>
void TimedSort(
	DoubleBuffer 						&device_storage,
	SizeT 									num_elements,
	typename DoubleBuffer::KeyType 		*h_keys,
	int 									g_iterations)
{
	typename DoubleBuffer::KeyType K;

	// Create sorting enactor
	radix_sort::Enactor sorting_enactor;

	// Move a fresh copy of the problem into device storage
	if (util::B40CPerror(cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice),
		"TimedSort cudaMemcpy device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);

	// Marker kernel in profiling stream
	util::FlushKernel<void><<<1,1>>>();

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	sorting_enactor.ENACTOR_DEBUG = true;
	sorting_enactor.template Sort<GENRE>(device_storage, num_elements, g_max_ctas);
	sorting_enactor.ENACTOR_DEBUG = false;

	// Perform the timed number of sorting g_iterations
	GpuTimer timer;

	double elapsed = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Move a fresh copy of the problem into device storage
		if (util::B40CPerror(cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice),
			"TimedSort cudaMemcpy device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);

		// Marker kernel in profiling stream
		util::FlushKernel<void><<<1,1>>>();

		// Start cuda timing record
		timer.Start();

		// Call the sorting API routine
		sorting_enactor.template Sort<GENRE>(device_storage, num_elements, g_max_ctas);

		// End cuda timing record
		timer.Stop();
		elapsed += (double) timer.ElapsedMillis();
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // Copy out data
    if (util::B40CPerror(cudaMemcpy(h_keys, device_storage.d_keys[device_storage.selector], sizeof(K) * num_elements, cudaMemcpyDeviceToHost),
		"TimedSort cudaMemcpy device_storage.d_keys failed: ", __FILE__, __LINE__)) exit(1);
}


/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of K elements, values of V elements, and then dispatches the problem 
 * to the GPU for the given number of g_iterations, displaying runtime information.
 *
 * @param[in] 		g_iterations
 * 		Number of times to invoke the GPU sorting primitive
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 */
template<
	typename K,
	typename V,
	typename SizeT>
void TestSort(SizeT num_elements)
{
    // Allocate the sorting problem on the host and fill the keys with random bytes

	K *h_keys = (K*) malloc(num_elements * sizeof(K));
	K *h_reference_keys = (K*) malloc(num_elements * sizeof(K));
	V *h_values = (g_keys_only) ?
		NULL :
		h_values = (V*) malloc(num_elements * sizeof(V));

	// Use random bits
	for (unsigned int i = 0; i < num_elements; ++i) {
		util::RandomBits<K>(h_keys[i], 0);
		h_reference_keys[i] = h_keys[i];
	}

	// Run the timing test
	if (g_keys_only) {

		printf("Keys-only, %d iterations, %d elements", g_iterations, num_elements);
		fflush(stdout);

		// Allocate device storage
		util::DoubleBuffer<K> device_storage;
		if (util::B40CPerror(cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements),
			"TimedSort cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);

		TimedSort<radix_sort::LARGE_SIZE>(
			device_storage, num_elements, h_keys, g_iterations);

	    // Free allocated memory
	    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
	    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);

	} else {

		printf("Key-values, %d iterations, %d elements", g_iterations, num_elements);
		fflush(stdout);

		// Allocate device storage
		util::DoubleBuffer<K, V> device_storage;
		if (util::B40CPerror(cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements),
			"TimedSort cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * num_elements),
			"TimedSort cudaMalloc device_storage.d_values[0] failed: ", __FILE__, __LINE__)) exit(1);

		TimedSort<radix_sort::LARGE_SIZE>(device_storage, num_elements, h_keys, g_iterations);

	    // Free allocated memory
	    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
	    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
	    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
	    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);
	}

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();
    
	// Display sorted key data
	if (g_verbose) {
		printf("\n\nKeys:\n");
		for (int i = 0; i < num_elements; i++) {	
			PrintValue<K>(h_keys[i]);
			printf(", ");
		}
		printf("\n\n");
	}	
	
    // Verify solution
	std::sort(h_reference_keys, h_reference_keys + num_elements);	
	CompareResults<K>(h_keys, h_reference_keys, num_elements, true);
	printf("\n");
	fflush(stdout);

	// Free our allocated host memory 
	if (h_keys != NULL) free(h_keys);
    if (h_values != NULL) free(h_values);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{
	// Initialize commandline args and device
	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	// Seed random number generator
	srand(0);				// presently deterministic
	//srand(time(NULL));

	// Use 32-bit integer for array indexing
	typedef int SizeT;
	SizeT num_elements = 1024;

	// Parse command line arguments
    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
    g_keys_only = args.CheckCmdLineFlag("keys-only");
	g_verbose = args.CheckCmdLineFlag("v");

	// Execute test(s)
/*	
	TestSort<float, float>(num_elements);

	TestSort<double, double>(num_elements);

	TestSort<char, char>(num_elements);

	TestSort<unsigned char, unsigned char>(num_elements);

	TestSort<short, short>(num_elements);

	TestSort<unsigned short, unsigned short>(num_elements);

	TestSort<int, int>(

	TestSort<unsigned int, unsigned int>(num_elements);

	TestSort<long long, long long>(num_elements);

	TestSort<unsigned long long, unsigned long long>(num_elements);

	TestSort<float, Fribbitz>(num_elements);
*/

	TestSort<unsigned int, unsigned int>(num_elements);

}



