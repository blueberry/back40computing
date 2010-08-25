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
 * Simple test driver program for SRTS Radix Sorting.
 *
 * Useful for demonstrating how to integrate SRTS Radix Sorting into your 
 * application
 ******************************************************************************/

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>
#include <algorithm>

#include <radixsort_api.cu>			// Sorting includes
#include <test_utils.cu>			// Utilities and correctness-checking
#include <cutil.h>					// Utilities for commandline parsing

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;


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
	printf("\nsrts_radix_sort [--device=<device index>] [--v] [--i=<num-iterations>] [--n=<num-elements>] [--keys-only]\n"); 
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
	unsigned int num_elements, 
	K *h_keys,
	unsigned int iterations)
{
	printf("Keys-only, %d iterations, %d elements", iterations, num_elements);
	
	//
	// Allocate device storage and create sorting enactor  
	//

	RadixSortStorage<K> device_storage;	
	CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.d_keys, sizeof(K) * num_elements) );

	RadixSortingEnactor<K> sorting_enactor(num_elements);


	
	//
	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	//
	
	CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice) );		// copy keys
	sorting_enactor.EnactSort(device_storage);
	
	
	//
	// Perform the timed number of sorting iterations
	//
	
	// Create timing records
	cudaEvent_t start_event, stop_event;
	CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
	CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		// Move a fresh copy of the problem into device storage
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice) );	// copy keys

		// Start cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

		// Call the sorting API routine
		sorting_enactor.EnactSort(device_storage);

		// End cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
		CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
		CUDA_SAFE_CALL( cudaEventElapsedTime(&duration, start_event, stop_event));
		elapsed += (double) duration;		
	}

	//
	// Display timing information
	//
	
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // 
    // Copy out data & free allocated memory
    //
    
    device_storage.CleanupTempStorage();						// clean up sort-allocated storage 

    CUDA_SAFE_CALL( cudaMemcpy(h_keys, device_storage.d_keys, sizeof(K) * num_elements, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree(device_storage.d_keys) );		// clean up keys

    // Clean up events
	CUDA_SAFE_CALL( cudaEventDestroy(start_event) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop_event) );
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
	unsigned int num_elements, 
	K *h_keys,
	V *h_values, 
	unsigned int iterations) 
{
	printf("Key-values, %d iterations, %d elements", iterations, num_elements);
	
	//
	// Allocate device storage and create sorting enactor  
	//

	RadixSortStorage<K, V> device_storage;	
	CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.d_keys, sizeof(K) * num_elements) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.d_values, sizeof(V) * num_elements) );

	RadixSortingEnactor<K, V> sorting_enactor(num_elements);

	
	//
	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	//
	
	CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice) );			// copy keys
	CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_values, h_values, sizeof(V) * num_elements, cudaMemcpyHostToDevice) );		// copy values
	sorting_enactor.EnactSort(device_storage);
	
	
	//
	// Perform the timed number of sorting iterations
	//
	
	// Create timing records
	cudaEvent_t start_event, stop_event;
	CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
	CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		// Move a fresh copy of the problem into device storage
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(K) * num_elements, cudaMemcpyHostToDevice) );			// copy keys
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_values, h_values, sizeof(V) * num_elements, cudaMemcpyHostToDevice) );		// copy values

		// Start cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

		// Call the sorting API routine
		sorting_enactor.EnactSort(device_storage);

		// End cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
		CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
		CUDA_SAFE_CALL( cudaEventElapsedTime(&duration, start_event, stop_event));
		elapsed += (double) duration;		
	}

	//
	// Display timing information
	//
	
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // 
    // Copy out data & free allocated memory
    //
    
    device_storage.CleanupTempStorage();						// clean up sort-allocated storage 

    CUDA_SAFE_CALL( cudaMemcpy(h_keys, device_storage.d_keys, sizeof(K) * num_elements, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(h_values, device_storage.d_values, sizeof(V) * num_elements, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree(device_storage.d_keys) );			// clean up keys
	CUDA_SAFE_CALL( cudaFree(device_storage.d_values) );		// clean up values

    // Clean up events
	CUDA_SAFE_CALL( cudaEventDestroy(start_event) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop_event) );
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
	unsigned int iterations,
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
	VerifySort<K>(h_keys, h_reference_keys, num_elements, true);
	printf("\n");
	fflush(stdout);

	// Free our allocated host memory 
	if (h_keys != NULL) free(h_keys);
    if (h_values != NULL) free(h_values);
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
    bool keys_only;

    //
	// Check command line arguments
    //

    if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 0;
	}

    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", (int*)&iterations);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);
    keys_only = cutCheckCmdLineFlag( argc, (const char**) argv, "keys-only");
	g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");

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
*/			
	TestSort<unsigned int, unsigned int>(
			iterations,
			num_elements, 
			keys_only);
/*	
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
}



