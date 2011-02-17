/******************************************************************************
 * Copyright 2010 Duane Merrill
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
 * 
 * 
 * 
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
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

// Sorting includes
#include "radixsort_api_enactor_tuned.cuh"
#include "radixsort_api_storage.cuh"

// Test utils
#include "b40c_util.h"					// Misc. utils (random-number gen, I/O, etc.)


// #define DEBUG

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;
int g_max_ctas = 0;


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
	printf("\ntest_large_problem_sorting [--device=<device index>] [--v] [--i=<num-iterations>] "
			"[--max-ctas=<max-thread-blocks>] [--n=<num-elements>] [--keys-only]\n");
	printf("\n");
	printf("\t--v\tDisplays sorted results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sorting operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
	printf("\t--keys-only\tSpecifies that keys are not accommodated by value pairings\n");
	printf("\n");
}



/**
 * Keys-only sorting.  Uses the GPU to sort the specified vector of elements for the given 
 * number of iterations, displaying runtime information.
 *
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		h_keys 
 * 		Vector of keys to sort 
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 */
template <typename K>
void TimedSort(
	int num_elements,
	K *h_keys,
	int iterations)
{
	printf("Keys-only, %d iterations, %d elements", iterations, num_elements);
	
	// Allocate device storage  
	MultiCtaSortStorage<K> device_storage(num_elements);		
	if (B40CPerror(cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements),
		"TimedSort cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);

	// Create sorting enactor
	LsbSortEnactorTuned sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	if (B40CPerror(cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice),
		"TimedSort cudaMemcpy device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);
	sorting_enactor.DEBUG = true;
	sorting_enactor.EnactSort(device_storage, g_max_ctas);
	sorting_enactor.DEBUG = false;

	// Perform the timed number of sorting iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		// Move a fresh copy of the problem into device storage
		if (B40CPerror(cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice),
			"TimedSort cudaMemcpy device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the sorting API routine
		sorting_enactor.EnactSort(device_storage, g_max_ctas);

		// End cuda timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;		
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Copy out data
    if (B40CPerror(cudaMemcpy(h_keys, device_storage.d_keys[device_storage.selector], sizeof(K) * num_elements, cudaMemcpyDeviceToHost),
		"TimedSort cudaMemcpy device_storage.d_keys failed: ", __FILE__, __LINE__)) exit(1);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
}



/**
 * Key-value sorting.  Uses the GPU to sort the specified vector of elements for the given 
 * number of iterations, displaying runtime information.
 *
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		h_keys 
 * 		Vector of keys to sort 
 * @param[in,out] 	h_values  
 * 		Vector of values to sort 
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 */
template <typename K, typename V>
void TimedSort(
	int num_elements,
	K *h_keys,
	V *h_values, 
	int iterations)
{
	printf("Key-values, %d iterations, %d elements", iterations, num_elements);
	
	// Allocate device storage   
	MultiCtaSortStorage<K, V> device_storage(num_elements);	
	if (B40CPerror(cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements),
		"TimedSort cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);
	if (B40CPerror(cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * num_elements),
		"TimedSort cudaMalloc device_storage.d_values[0] failed: ", __FILE__, __LINE__)) exit(1);

	// Create sorting enactor
	LsbSortEnactorTuned sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	if (B40CPerror(cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice),
		"TimedSort cudaMemcpy device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);
	sorting_enactor.DEBUG = true;
	sorting_enactor.EnactSort(device_storage, g_max_ctas);
	sorting_enactor.DEBUG = false;

	// Perform the timed number of sorting iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		// Move a fresh copy of the problem into device storage
		if (B40CPerror(cudaMemcpy(device_storage.d_keys[0], h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice),
			"TimedSort cudaMemcpy device_storage.d_keys[0] failed: ", __FILE__, __LINE__)) exit(1);

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the sorting API routine
		sorting_enactor.EnactSort(device_storage, g_max_ctas);

		// End cuda timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;		
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);

    // Copy out data 
    if (B40CPerror(cudaMemcpy(h_keys, device_storage.d_keys[device_storage.selector], sizeof(K) * num_elements, cudaMemcpyDeviceToHost),
		"TimedSort cudaMemcpy device_storage.d_keys failed: ", __FILE__, __LINE__)) exit(1);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);
}


/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of K elements, values of V elements, and then dispatches the problem 
 * to the GPU for the given number of iterations, displaying runtime information.
 *
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 */
template<typename K, typename V>
void TestSort(
	int iterations,
	int num_elements,
	bool keys_only)
{
    // Allocate the sorting problem on the host and fill the keys with random bytes

	K *h_keys = NULL;
	K *h_reference_keys = NULL;
	V *h_values = NULL;
	h_keys = (K*) malloc(num_elements * sizeof(K));
	h_reference_keys = (K*) malloc(num_elements * sizeof(K));
	if (!keys_only) h_values = (V*) malloc(num_elements * sizeof(V));

	// Use random bits
	for (unsigned int i = 0; i < num_elements; ++i) {
		RandomBits<K>(h_keys[i], 0);
		h_reference_keys[i] = h_keys[i];
	}

    // Run the timing test
	if (keys_only) {
		TimedSort<K>(num_elements, h_keys, iterations);
	} else {
		TimedSort<K, V>(num_elements, h_keys, h_values, iterations);
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

	CommandLineArgs args(argc, argv);
	DeviceInit(args);

	//srand(time(NULL));	
	srand(0);				// presently deterministic

    int num_elements 					= 1024;
    int iterations  					= 1;
    bool keys_only;

    //
	// Check command line arguments
    //

    if (args.CheckCmdLineFlag("help")) {
		Usage();
		return 0;
	}

    args.GetCmdLineArgumenti("i", iterations);
    args.GetCmdLineArgumenti("n", num_elements);
    args.GetCmdLineArgumenti("max-ctas", g_max_ctas);
    keys_only = args.CheckCmdLineFlag("keys-only");
	g_verbose = args.CheckCmdLineFlag("v");

/*	
	// Execute test(s)
	TestSort<float, float>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<double, double>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<char, char>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<unsigned char, unsigned char>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<short, short>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<unsigned short, unsigned short>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<int, int>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<unsigned int, unsigned int>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<long long, long long>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<unsigned long long, unsigned long long>(
			iterations,
			num_elements, 
			keys_only);
	TestSort<float, Fribbitz>(
			iterations,
			num_elements, 
			keys_only);
*/

	TestSort<unsigned int, unsigned int>(
			iterations,
			num_elements, 
			keys_only);

}



