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
 * 		Duane Merrill and Andrew Grimshaw, "Revisiting Sorting for GPGPU 
 * 		Stream Architectures," University of Virginia, Department of 
 * 		Computer Science, Charlottesville, VA, USA, Technical Report 
 * 		CS2010-03, 2010.
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 */


#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include <inc/cutil.h>


//------------------------------------------------------------------------------
// Sorting includes
//------------------------------------------------------------------------------

// Sorting includes
#include <srts_radix_sort.cu>

// Utilities and correctness-checking
#include <srts_verifier.cu>


//------------------------------------------------------------------------------
// Defines, constants, globals 
//------------------------------------------------------------------------------

unsigned int g_timer;

bool g_verbose;
bool g_verbose2;
bool g_verify;
int  g_entropy_reduction = 0;


//------------------------------------------------------------------------------
// Routines
//------------------------------------------------------------------------------


/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\nsrts_radix_sort [--noprompt] [--v[2]] [--i=<num-iterations>] [--n=<num-elements>] [--keys-only]\n"); 
	printf("\n");
	printf("\t--v\tDisplays kernel launch config info.\n");
	printf("\n");
	printf("\t--v2\tSame as --v, but displays the results to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sorting operation <num-iterations> times\n");
	printf("\t\t\ton the device.  (Only copies input/output once.) Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\t--keys-only\tSpecifies that keys are not accommodated by value pairings\n");
	printf("\n");
	printf("[--entropy-reduction=<level>\tSpecifies the number of bitwise-AND'ing\n");
	printf("\t\t\titerations for random key data.  Default = 0, Identical keys = -1\n");
	printf("\n");
}


/**
 * Generates random 32-bit keys.
 * 
 * We always take the second-order byte from rand() because the higher-order 
 * bits returned by rand() are commonly considered more uniformly distributed
 * than the lower-order bits.
 * 
 * We can decrease the entropy level of keys by adopting the technique 
 * of Thearling and Smith in which keys are computed from the bitwise AND of 
 * multiple random samples: 
 * 
 * entropy_reduction	| Effectively-unique bits per key
 * -----------------------------------------------------
 * -1					| 0
 * 0					| 32
 * 1					| 25.95
 * 2					| 17.41
 * 3					| 10.78
 * 4					| 6.42
 * ...					| ...
 * 
 */
template <typename K>
void RandomBits(K &key, int entropy_reduction) 
{
	const unsigned int NUM_USHORTS = (sizeof(K) + sizeof(unsigned short) - 1) / sizeof(unsigned short);
	unsigned short key_bits[NUM_USHORTS];
	
	for (int j = 0; j < NUM_USHORTS; j++) {
		unsigned short halfword = 0xffff; 
		for (int i = 0; i <= entropy_reduction; i++) {
			halfword &= (rand() >> 8);
		}
		key_bits[j] = halfword;
	}
		
	memcpy(&key, key_bits, sizeof(K));
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
	unsigned int iterations
) 
{
	//
	// Create device storage for the sorting problem
	//

    GlobalStorage<K, V> device_storage = {NULL, NULL, NULL, NULL, NULL};

	// Allocate and initialize device memory for keys
	unsigned int keys_mem_size = sizeof(K) * num_elements;
	CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.keys, keys_mem_size) );

	// Allocate and initialize device memory for data
	unsigned int data_mem_size = sizeof(V) * num_elements;
	if (h_data != NULL) {
		CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.data, data_mem_size) );
	}
	
	
	//
	// Perform a single iteration to allocate memory, prime code caches, etc.
	//
	
	CUDA_SAFE_CALL( cudaMemcpy(device_storage.keys, h_keys, keys_mem_size, cudaMemcpyHostToDevice) );
	if (h_data != NULL) CUDA_SAFE_CALL( cudaMemcpy(device_storage.data, h_data, data_mem_size, cudaMemcpyHostToDevice) );

	LaunchSort<K, V>(
		num_elements, 
		device_storage,
		g_verbose);

	
	//
	// Perform the timed number of sorting iterations
	//
	
	// Create timing records
	cudaEvent_t start_event, stop_event;
	CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
	CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

	// Make sure there are no CUDA errors before we launch
	CUT_CHECK_ERROR("Kernel execution failed (errors before launch)");
	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		// Move a fresh copy of the problem into device storage
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.keys, h_keys, keys_mem_size, cudaMemcpyHostToDevice) );
		if (h_data != NULL) CUDA_SAFE_CALL( cudaMemcpy(device_storage.data, h_data, data_mem_size, cudaMemcpyHostToDevice) );

		// Start cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

		LaunchSort<K, V>(
			num_elements, 
			device_storage,
			false);

		// End cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
		CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
		CUDA_SAFE_CALL( cudaEventElapsedTime(&duration, start_event, stop_event));
		elapsed += (double) duration;		
	}

	//
	// Display timing information
	//
	
	// Display 
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf("%s, %d entropy reduction, %d iterations, %d elements, %f GPU ms, %f x10^9 elts/sec\n", 
		(h_data == NULL) ? "keys-only" : "key-value",
		g_entropy_reduction,
		iterations, 
		num_elements,
		avg_runtime,
		throughput);
    
	
    // 
    // Copy out data & free allocated memory
    //
    
	// Sorted keys 
	CUDA_SAFE_CALL( cudaMemcpy(h_keys, device_storage.keys, keys_mem_size, cudaMemcpyDeviceToHost) );
    CUDA_SAFE_CALL( cudaFree(device_storage.keys) );
    CUDA_SAFE_CALL( cudaFree(device_storage.temp_keys) );

	// Sorted values 
	if (h_data != NULL) {
		CUDA_SAFE_CALL( cudaMemcpy(h_data, device_storage.data, data_mem_size, cudaMemcpyDeviceToHost) );
	    CUDA_SAFE_CALL( cudaFree(device_storage.data) );
	    CUDA_SAFE_CALL( cudaFree(device_storage.temp_data) );
	}

	// Free spine storage
    CUDA_SAFE_CALL( cudaFree(device_storage.temp_spine) );

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

	for (unsigned int i = 0; i < num_elements; ++i) {
		RandomBits<K>(h_keys[i], g_entropy_reduction);
	}
	
    // Run the timing test 

	TimedSort<K, V>(num_elements, h_keys, h_data, iterations);
    
    // Verify solution

	if (g_verify) VerifySort<K>(h_keys, num_elements, g_verbose);
	printf("\n");
	fflush(stdout);
	
	// Display sorted key data

	if (g_verbose2) {
		printf("\n\nKeys:\n");
		for (int i = 0; i < num_elements; i++) {	
			PrintValue<K>(h_keys[i]);
			printf(", ");
		}
		printf("\n");
	}	
	
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
	cutGetCmdLineArgumenti( argc, (const char**) argv, "entropy-reduction", (int*)&g_entropy_reduction);
    keys_only = cutCheckCmdLineFlag( argc, (const char**) argv, "keys-only");
	if (g_verbose2 = cutCheckCmdLineFlag( argc, (const char**) argv, "v2")) {
		g_verbose = true;
	} else {
		g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");
	}
	g_verify = !cutCheckCmdLineFlag( argc, (const char**) argv, "noverify");
	
	// Execute test
//	TestSort<float, float>(
//	TestSort<double, double>(
//	TestSort<char, char>(
//	TestSort<unsigned char, unsigned char>(
//	TestSort<short, short>(
//	TestSort<unsigned short, unsigned short>(
//	TestSort<int, int>(
	TestSort<unsigned int, unsigned int>(
//	TestSort<long long, long long>(
//	TestSort<unsigned long long, unsigned long long>(
		iterations,
		num_elements, 
		keys_only);

	CUT_EXIT(argc, argv);
}



