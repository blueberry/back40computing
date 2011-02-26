/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * Simple test driver program for *large-problem* reduction.
 ******************************************************************************/

#include <stdio.h> 

// Reduction includes
#include "reduction_api_granularity.cuh"
#include "reduction_api_enactor_tuned.cuh"

// Test utils
#include "b40c_util.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


using namespace b40c;
using namespace reduction;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool 	g_verbose 						= false;
int 	g_max_ctas 						= 0;
int 	g_iterations  					= 1;


/******************************************************************************
 * Test wrappers for binary, associative operations
 ******************************************************************************/

template <typename T>
struct Sum
{
	static __host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return a + b;
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return 0;
	}
};

template <typename T>
struct Max
{
	static __host__ __device__ __forceinline__ T Op(const T &a, const T &b)
	{
		return (a > b) ? a : b;
	}

	static __host__ __device__ __forceinline__ T Identity()
	{
		return 0;
	}
};


/******************************************************************************
 * Utility Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_reduction [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--sweep]\n");
	printf("\n");
	printf("\t--v\tDisplays copied results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the reduction operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of bytes to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}


/**
 * Timed reduction.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity()>
double TimedReduction(
	T *h_data,
	T *h_reference,
	size_t num_elements)
{
	T h_dest[1] = {0};
	printf("B40C Reduction: %d iterations, %d elements, ", g_iterations, num_elements);
	
	// Allocate device storage  
	T *d_src, *d_dest;
	if (B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedReduction cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T)),
		"TimedReduction cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Create enactor
	ReductionEnactorTuned reduction_enactor;

	// Move a fresh copy of the problem into device storage
	if (B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedReduction cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);
	
	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	reduction_enactor.template Enact<T, BinaryOp, Identity>(
		d_dest, d_src, num_elements, g_max_ctas);

	// Perform the timed number of iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start timing record
		cudaEventRecord(start_event, 0);

		// Call the reduction API routine
		reduction_enactor.template Enact<T, BinaryOp, Identity>(
			d_dest, d_src, num_elements, g_max_ctas);

		// End timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;		
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, throughput * sizeof(T));
	
    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Copy out data
    if (B40CPerror(cudaMemcpy(h_dest, d_dest, sizeof(T), cudaMemcpyDeviceToHost),
		"TimedReduction cudaMemcpy d_dest failed: ", __FILE__, __LINE__)) exit(1);

    // Free allocated memory
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Display copied data
	if (g_verbose) {
		printf("Reduction: ");
		PrintValue(h_dest[0]);
		printf(", Reference: ");
		PrintValue(h_reference[0]);
		printf("\n\n");
	}

    // Verify solution
	CompareResults(h_dest, h_reference, 1, true);
	printf("\n");
	fflush(stdout);

	return throughput;
}


/**
 * Timed reduction.  Uses the GPU to copy the specified vector of elements for the given
 * number of iterations, displaying runtime information.
 */
template <
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity()>
double TimedThrustReduction(
	T *h_data,
	T *h_reference,
	size_t num_elements)
{
	T h_dest[1] = {0};
	printf("Thrust Reduction: %d iterations, %d elements, ", g_iterations, num_elements);
	
	// Allocate device storage  
	T *d_src, *d_dest;
	if (B40CPerror(cudaMalloc((void**) &d_src, sizeof(T) * num_elements),
		"TimedReduction cudaMalloc d_src failed: ", __FILE__, __LINE__)) exit(1);
	if (B40CPerror(cudaMalloc((void**) &d_dest, sizeof(T)),
		"TimedReduction cudaMalloc d_dest failed: ", __FILE__, __LINE__)) exit(1);

	// Move a fresh copy of the problem into device storage
	if (B40CPerror(cudaMemcpy(d_src, h_data, sizeof(T) * num_elements, cudaMemcpyHostToDevice),
		"TimedReduction cudaMemcpy d_src failed: ", __FILE__, __LINE__)) exit(1);
	
	// Perform a single iteration to allocate any memory if needed, prime code caches, etc.
	thrust::device_ptr<T> dev_ptr(d_src);		
	h_dest[0] = thrust::reduce(dev_ptr, dev_ptr + num_elements, (T) 0, thrust::plus<T>());
	
	// Perform the timed number of iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < g_iterations; i++) {

		// Start timing record
		cudaEventRecord(start_event, 0);

		h_dest[0] = thrust::reduce(dev_ptr, dev_ptr + num_elements, (T) 0, thrust::plus<T>());
		
		// End timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;		
	}

	// Display timing information
	double avg_runtime = elapsed / g_iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0;
    printf("%f GPU ms, %f x10^9 elts/sec, %f x10^9 B/sec, ",
		avg_runtime, throughput, throughput * sizeof(T));
	
    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Free allocated memory
    if (d_src) cudaFree(d_src);
    if (d_dest) cudaFree(d_dest);

	// Flushes any stdio from the GPU
	cudaThreadSynchronize();

	// Display copied data
	if (g_verbose) {
		printf("Reduction: ");
		PrintValue(h_dest[0]);
		printf(", Reference: ");
		PrintValue(h_reference[0]);
		printf("\n\n");
	}

    // Verify solution
	CompareResults(h_dest, h_reference, 1, true);
	printf("\n");
	fflush(stdout);

	return throughput;
}





/**
 * Creates an example reduction problem and then dispatches the problem
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity()>
void TestReduction(size_t num_elements)
{
    // Allocate the reduction problem on the host and fill the keys with random bytes

	T *h_data 			= (T*) malloc(num_elements * sizeof(T));
	T *h_reference 		= (T*) malloc(sizeof(T));

	if ((h_data == NULL) || (h_reference == NULL)){
		fprintf(stderr, "Host malloc of problem data failed\n");
		exit(1);
	}

	// Identity
	h_reference[0] = Identity();

	for (size_t i = 0; i < num_elements; ++i) {
		// RandomBits<T>(h_data[i], 0);
		h_data[i] = i;
		h_reference[0] = BinaryOp(h_reference[0], h_data[i]);
	}

	//
    // Run the timing test(s)
	//
	double b40c = TimedReduction<T, BinaryOp, Identity>(h_data, h_reference, num_elements);
	double thrust = TimedThrustReduction<T, BinaryOp, Identity>(h_data, h_reference, num_elements);
	printf("B40C speedup: %.2f\n", b40c/thrust);
	

	// Free our allocated host memory 
	if (h_data) free(h_data);
    if (h_reference) free(h_reference);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main(int argc, char** argv)
{

	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//srand(time(NULL));	
	srand(0);				// presently deterministic

    //
	// Check command line arguments
    //

	size_t num_elements = 1024;

    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    bool sweep = args.CheckCmdLineFlag("sweep");
    args.GetCmdLineArgument("i", g_iterations);
    args.GetCmdLineArgument("n", num_elements);
    args.GetCmdLineArgument("max-ctas", g_max_ctas);
	g_verbose = args.CheckCmdLineFlag("v");

	// Execute test(s), optionally sweeping problem size downward
	{
		printf("\n-- UNSIGNED CHAR ----------------------------------------------\n");
		typedef unsigned char T;
		typedef Sum<T> BinaryOp;
		TestReduction<T, BinaryOp::Op, BinaryOp::Identity>(num_elements * 4);
	}
	{
		printf("\n-- UNSIGNED SHORT ----------------------------------------------\n");
		typedef unsigned short T;
		typedef Sum<T> BinaryOp;
		TestReduction<T, BinaryOp::Op, BinaryOp::Identity>(num_elements * 2);
	}
	{
		printf("\n-- UNSIGNED INT -----------------------------------------------\n");
		typedef unsigned int T;
		typedef Sum<T> BinaryOp;
		TestReduction<T, BinaryOp::Op, BinaryOp::Identity>(num_elements);
	}
	{
		printf("\n-- UNSIGNED LONG LONG -----------------------------------------\n");
		typedef unsigned long long T;
		typedef Sum<T> BinaryOp;
		TestReduction<T, BinaryOp::Op, BinaryOp::Identity>(num_elements / 2);
	}
	
	
	
/*	
    size_t orig_num_elements = num_elements;
    do {

    	TestReduction<T, BinaryOp::Op, BinaryOp::Identity>(num_elements);
    	num_elements -= 4096;

    } while (sweep && (num_elements < orig_num_elements ));
*/    

	return 0;
}



