/**
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
 */


//------------------------------------------------------------------------------
// Simple test driver program for SRTS Radix Sorting.
//
// Useful for demonstrating how to integrate SRTS Radix Sorting into your 
// application
//------------------------------------------------------------------------------

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include <inc/cutil.h>


//------------------------------------------------------------------------------
// Sorting includes
//------------------------------------------------------------------------------

#include <srts_radix_sort.cu>			// Sorting includes
#include <test_radix_sort_utils.cu>		// Utilities and correctness-checking


//------------------------------------------------------------------------------
// Defines, constants, globals 
//------------------------------------------------------------------------------

bool g_verbose;


//------------------------------------------------------------------------------
// Test structures
//------------------------------------------------------------------------------

// Test value-type structure 
struct Fribbitz {
	char a;
	double b;
	unsigned short c;
};


//------------------------------------------------------------------------------
// Routines
//------------------------------------------------------------------------------

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
 * Uses the GPU to sort the specified vector of elements for the given 
 * number of iterations, displaying runtime information.
 *
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		h_keys 
 * 		Vector of keys to sort 
 * @param[in,out] 	h_data  
 * 		Vector of values to sort (may be null)
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 */
template <typename K, typename V>
void TimedSort(
	unsigned int num_elements, 
	K *h_keys,
	V *h_data, 
	unsigned int iterations,
	bool keys_only) 
{
    printf("%s, %d iterations, %d elements", 
		(h_data == NULL) ? "keys-only" : "key-value",
		iterations, 
		num_elements);
    fflush(stdout);
	
	//
	// Create device storage for the sorting problem
	//

    GlobalStorage<K, V> 	key_value_storage = {NULL, NULL, NULL, NULL, NULL};
    GlobalStorage<K> 		keys_only_storage = {NULL, NULL, NULL, NULL, NULL};

	// Allocate and initialize device memory for keys
	unsigned int keys_mem_size = sizeof(K) * num_elements;
	CUDA_SAFE_CALL( cudaMalloc((void**) &key_value_storage.keys, keys_mem_size) );
	keys_only_storage.keys = key_value_storage.keys;

	// Allocate and initialize device memory for data
	unsigned int data_mem_size = sizeof(V) * num_elements;
	if (!keys_only) {
		CUDA_SAFE_CALL( cudaMalloc((void**) &key_value_storage.data, data_mem_size) );
	}
	
	CUT_CHECK_ERROR("Kernel execution failed (errors before launch)");
	
	
	//
	// Perform a single iteration to allocate memory, prime code caches, etc.
	//
	CUDA_SAFE_CALL( cudaMemcpy(key_value_storage.keys, h_keys, keys_mem_size, cudaMemcpyHostToDevice) );	// copy keys
	if (keys_only) { 
		LaunchKeysOnlySort<K>(num_elements, keys_only_storage);
	} else {
		CUDA_SAFE_CALL( cudaMemcpy(key_value_storage.data, h_data, data_mem_size, cudaMemcpyHostToDevice) );
		LaunchKeyValueSort<K, V>(num_elements, key_value_storage);
	}
	
	
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
		CUDA_SAFE_CALL( cudaMemcpy(key_value_storage.keys, h_keys, keys_mem_size, cudaMemcpyHostToDevice) );
		if (h_data != NULL) CUDA_SAFE_CALL( cudaMemcpy(key_value_storage.data, h_data, data_mem_size, cudaMemcpyHostToDevice) );

		// Start cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

		// Call the sorting API routine
		if (keys_only) 
			LaunchKeysOnlySort<K>(num_elements, keys_only_storage);
		else 
			LaunchKeyValueSort<K, V>(num_elements, key_value_storage);

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
    
	// Sorted keys 
	CUDA_SAFE_CALL( cudaMemcpy(h_keys, key_value_storage.keys, keys_mem_size, cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaFree(key_value_storage.keys) );
    CUDA_SAFE_CALL( cudaFree(key_value_storage.temp_keys) );

	// Sorted values 
	if (h_data != NULL) {
		CUDA_SAFE_CALL( cudaMemcpy(h_data, key_value_storage.data, data_mem_size, cudaMemcpyDeviceToHost) );
	    CUDA_SAFE_CALL( cudaFree(key_value_storage.data) );
	    CUDA_SAFE_CALL( cudaFree(key_value_storage.temp_data) );
	}

	// Free spine storage
    CUDA_SAFE_CALL( cudaFree(key_value_storage.temp_spine) );

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
	V *h_data = NULL;
	h_keys = (K*) malloc(num_elements * sizeof(K));
	if (!keys_only) h_data = (V*) malloc(num_elements * sizeof(V));

	// Use random bits
	for (unsigned int i = 0; i < num_elements; ++i) {
		RandomBits<K>(h_keys[i], 0);
	}

    // Run the timing test 
	TimedSort<K, V>(num_elements, h_keys, h_data, iterations, keys_only);
    
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
	VerifySort<K>(h_keys, num_elements, true);
	printf("\n");
	fflush(stdout);

	// Free our allocated host memory 
	if (h_keys != NULL) free(h_keys);
    if (h_data != NULL) free(h_data);
}


//------------------------------------------------------------------------------
// Main
//------------------------------------------------------------------------------

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
	
	// Execute test(s)
/*	
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



