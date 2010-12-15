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
 * Simple test driver program for *small-problem* radix sorting (with an 
 * optionally-reduced number of valid key-bits).
 *
 * Useful for demonstrating how to integrate LsbSingleGrid radix sorting into 
 * your application 
 ******************************************************************************/

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

// Sorting includes
#include <radixsort_single_grid.cu>
#include <radixsort_early_exit.cu>		

#include <test_utils.cu>				// Utilities and correctness-checking
#include <cutil.h>						// Utilities for commandline parsing
#include <b40c_util.h>					// Misc. utils (random-number gen, I/O, etc.)

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

//#define __B40C_ERROR_CHECKING__		 

bool g_verbose;


/******************************************************************************
 * Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\ntest_small_problem_sorting [--device=<device index>] [--v] [--i=<num-iterations>] [--n=<num-elements>] [--keys-only]\n"); 
	printf("\n");
	printf("\t--v\tDisplays sorted results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sorting operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
}



/**
 * Uses the small-problem-sorter (single-grid enactor) sort the specified 17-bit sorting
 * problem whose keys is a vector of the specified number of unsigned int elements, 
 * values of unsigned int elements.
 *
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		h_keys 
 * 		Vector of keys to sort 
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 */
template <typename K, typename V, int LOWER_KEY_BITS> 
void SmallProblemTimedSort(
	unsigned int num_elements, 
	K *h_keys,
	K *h_reference_keys,
	unsigned int iterations)
{
	printf("Single-kernel, small-problem key-value sort, %d iterations, %d elements", iterations, num_elements);
	
	// Allocate device storage   
	MultiCtaRadixSortStorage<K, V> device_storage(num_elements);	
	cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements);
    dbg_perror_exit("SmallProblemTimedSort:: cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__);
    cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * num_elements);
    dbg_perror_exit("SmallProblemTimedSort:: cudaMalloc device_storage.d_values[0] failed: ", __FILE__, __LINE__);

	// Create sorting enactor
	SingleGridRadixSortingEnactor<K, V> sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	cudaMemcpy(
		device_storage.d_keys[0], 
		h_keys, 
		sizeof(K) * num_elements, 
		cudaMemcpyHostToDevice);											// copy keys
	sorting_enactor.template EnactSort<LOWER_KEY_BITS>(device_storage);		// sort

	// Perform the timed number of sorting iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		sorting_enactor.RADIXSORT_DEBUG = (i == 0);

		// Move a fresh copy of the problem into device storage
		cudaMemcpy(
			device_storage.d_keys[0], 
			h_keys, 
			sizeof(K) * num_elements, 
			cudaMemcpyHostToDevice);										// copy keys

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the sorting API routine
		sorting_enactor.template EnactSort<LOWER_KEY_BITS>(device_storage);	// sort

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
	
    // Copy out keys 
    cudaMemcpy(
    	h_keys, 
    	device_storage.d_keys[device_storage.selector], 
    	sizeof(K) * num_elements, 
    	cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
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
	CompareResults<K>(h_keys, h_reference_keys, num_elements, true);
	printf("\n");
	fflush(stdout);
}



/**
 * Uses the large-problem-sorter (early-exit enactor) sort the specified sorting
 * problem whose keys is a vector of the specified number of K elements, 
 * values of V elements.
 *
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		h_keys 
 * 		Vector of keys to sort 
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 */
template <typename K, typename V, int LOWER_KEY_BITS> 
void LargeProblemTimedSort(
	unsigned int num_elements, 
	K *h_keys,
	K *h_reference_keys,
	unsigned int iterations)
{
	printf("Early-exit key-value sort, %d iterations, %d elements", iterations, num_elements);
	
	// Allocate device storage   
	MultiCtaRadixSortStorage<K, V> device_storage(num_elements);	
	cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements);
    dbg_perror_exit("LargeProblemTimedSort:: cudaMalloc device_storage.d_keys[0] failed: ", __FILE__, __LINE__);
	cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * num_elements);
    dbg_perror_exit("LargeProblemTimedSort:: cudaMalloc device_storage.d_values[0] failed: ", __FILE__, __LINE__);

	// Create sorting enactor
	EarlyExitRadixSortingEnactor<K, V> sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	cudaMemcpy(
		device_storage.d_keys[0], 
		h_keys, 
		sizeof(K) * num_elements, 
		cudaMemcpyHostToDevice);		// copy keys
	sorting_enactor.EnactSort(device_storage);

	// Perform the timed number of sorting iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		sorting_enactor.RADIXSORT_DEBUG = (i == 0);

		// Move a fresh copy of the problem into device storage
		cudaMemcpy(
			device_storage.d_keys[0], 
			h_keys, 
			sizeof(K) * num_elements, 
			cudaMemcpyHostToDevice);		// copy keys

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the sorting API routine
		sorting_enactor.EnactSort(device_storage);

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
	
    // Copy out keys 
    cudaMemcpy(
    	h_keys, 
    	device_storage.d_keys[device_storage.selector], 
    	sizeof(K) * num_elements, 
    	cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
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
	CompareResults<K>(h_keys, h_reference_keys, num_elements, true);
	printf("\n");
	fflush(stdout);
}


/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of elements having the specfied number of valid bits, and then 
 * dispatches the problem to the GPU for the given number of iterations, 
 * displaying runtime information.
 *
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in]		use_small_problem_enactor
 */
template <typename K, typename V, int LOWER_KEY_BITS> 
void TestSort(
	unsigned int iterations,
	int num_elements) 
{
    // Allocate the sorting problem on the host and fill the keys with random bytes

	K *h_keys = NULL;
	K *h_reference_keys = NULL;
	h_keys = (K*) malloc(num_elements * sizeof(K));
	h_reference_keys = (K*) malloc(num_elements * sizeof(K));

	// Use random bits
	for (unsigned int i = 0; i < num_elements; ++i) {
		RandomBits<K>(h_keys[i], 0, LOWER_KEY_BITS);
		h_reference_keys[i] = h_keys[i];
	}

	// Sort the reference keys
	std::sort(h_reference_keys, h_reference_keys + num_elements);	

	//
    // Run the timing tests
	//
	
	// Single-grid enactor (explicit passes)
	SmallProblemTimedSort<K, V, LOWER_KEY_BITS>(
		num_elements, h_keys, h_reference_keys, iterations);

	// Early-exit enactor (dynamic pass detection)
	LargeProblemTimedSort<K, V, LOWER_KEY_BITS>(
		num_elements, h_keys, h_reference_keys, iterations);

	// Free our allocated host memory 
	if (h_keys != NULL) free(h_keys);
	if (h_reference_keys != NULL) free(h_reference_keys);
}


/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv) {

	CUT_DEVICE_INIT(argc, argv);

	//srand(time(NULL));	
	srand(0);				// presently deterministic

    unsigned int num_elements 					= 1024;
    unsigned int iterations  					= 1;

	// Check command line arguments
    if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 0;
	}
    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", (int*)&iterations);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);
	g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");

	// Run sorting examples
	
	TestSort<unsigned int, unsigned int, 17>(			// only sort lower 17 bits 
		iterations, num_elements);

/*	
	TestSort<float, float, sizeof(float) * 8>(
		iterations, num_elements);	
	TestSort<long long, long long, sizeof(long long) * 8>(
		iterations, num_elements);
	TestSort<char, char, sizeof(char) * 8>(
		iterations, num_elements);
	TestSort<double, double, sizeof(double) * 8>(
		iterations, num_elements);
*/		
	
	cudaThreadSynchronize();
}



