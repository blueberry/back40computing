/******************************************************************************
 * 
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
 * 
 ******************************************************************************/


/******************************************************************************
 * 
 * Advanced test driver program for SRTS Radix Sorting
 *
 * WARNING: This program assumes knowlege of the temporary storage management
 * needed for performing SRTS radix sort -- do not use it as a reference for 
 * embedding SRTS sorting within your application.  See the Simple test driver 
 * program instead.
 * 
 ******************************************************************************/

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include <radixsort_api.cu>			// Sorting includes
#include "test_utils.cu"			// Utilities and correctness-checking
#include "inc/cutil.h"				// Utilities for commandline parsing

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;
bool g_verbose2;
bool g_verify;
int  g_entropy_reduction = 0;
bool g_regen;


/******************************************************************************
 * Empty Kernels
 ******************************************************************************/

/**
 * Dummy kernel to demarcate iterations of the same problem size in the profiler logs 
 */
__global__ void DummyKernel()
{
}



/******************************************************************************
 * Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\nsrts_radix_sort [--device=<device index>] [--v[2]] [--noverify]\n");
	printf("[--i=<num-iterations>] [--entropy-reduction=<level>] [--regen]\n");
	printf("[--key-bytes=<1|2|4|8>] [--value-bytes=<0|4|8|16>]\n");
	printf("[--n=<num-elements> | --n-input=<num-elements listfile>]\n");
	printf("[--max-blocks=<max-thread-blocks> | --max-blocks-input=<max-thread-blocks listfile>]\n");
	printf("\n");
	printf("\t--v\tDisplays kernel launch config info.\n");
	printf("\n");
	printf("\t--v2\tSame as --v, but displays the sorted keys to the console.\n");
	printf("\n");
	printf("\t--i\tPerforms the sorting operation <num-iterations> times\n");
	printf("\t\t\ton the device. Re-copies original input each time. Default = 1\n");
	printf("\n");
	printf("\t--n\tThe number of elements to comprise the sample problem\n");
	printf("\t\t\tDefault = 512\n");
	printf("\n");
	printf("\t--n-input\tA file of problem sizes, one per line.\n");
	printf("\n");
	printf("\t--max-blocks\tThe maximum number of threadblocks to launch.\n");
	printf("\t\t\tDefault = -1 (i.e., the API will select an appropriate value)\n");
	printf("\n");
	printf("\t--max-blocks-input\tA file of maximum threadblocks, one per line.\n");
	printf("\n");
	printf("\t--key-bytes\tThe number of key bytes to use.  Default=4\n");
	printf("\n");
	printf("\t--value-bytes\tThe number of value satellite bytes to pair with\n");
	printf("\t\t\tthe key.  Default=0 (I.e., keys-only)\n");
	printf("\n");
	printf("[\t--entropy-reduction=<level>\tSpecifies the number of bitwise-AND'ing\n");
	printf("\t\t\titerations for random key data.  Default = 0, Identical keys = -1\n");
	printf("\n");
	printf("[\t--regen\tGenerates new random numbers for every problem size \n");
	printf("\t\t\tin <num-elements-listfile>\n");
	printf("\n");
	printf("\t--noverify\tSpecifies that results should not be copied back and checked for correctness\n");
	printf("\n");
}


/**
 * Reads a newline-separated list of numbers from an input file.
 * Allocates memory for the returned list.
 */
void ReadList(
	int* &list, 
	unsigned int &len, 
	char* filename, 
	unsigned int default_val) 
{
	if (filename == NULL) {
		len = 1;
		list = (int*) malloc(len * sizeof(int));
		list[0] = default_val;
		return;
	}

	unsigned int data;
	FILE* fin = fopen(filename, "r");
	if (fin == NULL) {
		fprintf(stderr, "Could not open file.  Exiting.\n");
		exit(1);
	}
	len = 0;

	while(fscanf(fin, "%d\n", &data) > 0) {
		len++;
	}

	list = (int*) malloc(len * sizeof(int));
	rewind(fin);
	len = 0;

	while(fscanf(fin, "%d\n", &data) > 0) {
	
		list[len] = data;
		len++;
	}

	fclose(fin);
}




/**
 * Uses the GPU to sort the specified vector of elements for the given 
 * number of iterations, displaying runtime information.
 */
template <typename K, typename V, bool KEYS_ONLY>
void TimedSort(
	unsigned int num_elements, 
	unsigned int max_grid_size,
	K *h_keys,
	K *h_keys_result,
	RadixSortStorage<K, V>	&device_storage,
	unsigned int iterations) 
{
	// Create and initialize sorting enactor
	RadixSortingEnactor<K, V> sorting_enactor(num_elements, max_grid_size);

	CUT_CHECK_ERROR("Kernel execution failed (errors before launch)");

	// Create timing records
	cudaEvent_t start_event, stop_event;
	CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
	CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );

	// Perform the timed number of sorting iterations
	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		RADIXSORT_DEBUG = (g_verbose && (i == 0));

		// Move a fresh copy of the problem into device storage
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, num_elements * sizeof(K), cudaMemcpyHostToDevice) );

		// Start cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(start_event, 0) );

		// Call the sorting API routine
		cudaError_t retval = sorting_enactor.EnactSort(device_storage);

		// End cuda timing record
		CUDA_SAFE_CALL( cudaEventRecord(stop_event, 0) );
		CUDA_SAFE_CALL( cudaEventSynchronize(stop_event) );
		CUDA_SAFE_CALL( cudaEventElapsedTime(&duration, start_event, stop_event));
		elapsed += (double) duration;
		
		if (i == 0) {
			printf("%d-byte keys, %d-byte values, %d iterations, %d elements, %d max grid size", 
				sizeof(K), 
				(KEYS_ONLY) ? 0 : sizeof(V),
				iterations, 
				num_elements,
				max_grid_size);
			fflush(stdout);
		}
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);

    // Clean up events
	CUDA_SAFE_CALL( cudaEventDestroy(start_event) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop_event) );
	
	// Copy out sorted keys and check
    if (g_verify || g_verbose) {

    	CUDA_SAFE_CALL( cudaMemcpy(h_keys_result, device_storage.d_keys, num_elements * sizeof(K), cudaMemcpyDeviceToHost) );

		// Display sorted key data
		if (g_verbose2) {
			printf("\n\nKeys:\n");
			for (int i = 0; i < num_elements; i++) {	
				PrintValue<K>(h_keys_result[i]);
				printf(", ");
			}
			printf("\n\n");
		}	
		
	    // Verify solution
		if (g_verify) {
			VerifySort<K>(h_keys_result, num_elements, true);
			printf("\n");
			fflush(stdout);
		}
    }
	
}


/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of K elements, values of V elements, and then dispatches the problem 
 * to the GPU for the given number of iterations, displaying runtime information.
 */
template<typename K, typename V, bool KEYS_ONLY>
void TestSort(
	unsigned int iterations,
	int* num_elements_list,
	unsigned int num_elements_list_size,
	int* max_grid_sizes,
	unsigned int num_max_grid_sizes) 
{
	
	K *h_keys, *h_keys_result;
	
	// Find largest maximum grid size in list of maximum grid sizes
	int max_grid_size = -1;
	for (int i = 0; i < num_max_grid_sizes; i++) {
		if (max_grid_sizes[i] > max_grid_size) {
			max_grid_size = max_grid_sizes[i];
		}
	}

	// find maximum problem size in the list of problem sizes
	unsigned int max_num_elements = 0;
	unsigned int max_num_spine_elements = 0; 
	for (int i = 0; i < num_elements_list_size; i++) {

		RadixSortingEnactor<K, V> sorting_enactor(num_elements_list[i], max_grid_size);
		
		if (sorting_enactor.CanFit()) {
			if (num_elements_list[i] > max_num_elements) {
				max_num_elements = num_elements_list[i];
			}
			unsigned int num_spine_elements = sorting_enactor.SpineElements();
			if (num_spine_elements > max_num_spine_elements) {
				max_num_spine_elements = num_spine_elements;
			}
		}
	}
	
	// Allocate host memory
	h_keys = (K*) malloc(max_num_elements * sizeof(K));
	h_keys_result = (K*) malloc(max_num_elements * sizeof(K));

	// Randomly initialize the keyset on the host
	for (unsigned int j = 0; j < max_num_elements; j++) {
		RandomBits<K>(h_keys[j], g_entropy_reduction);
	}

	// Allocate device memory
	RadixSortStorage<K, V> device_storage;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_storage.d_keys, max_num_elements * sizeof(K)) );
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_storage.d_alt_keys, max_num_elements * sizeof(K)));
	if (!KEYS_ONLY) {
		CUDA_SAFE_CALL( cudaMalloc( (void**) &device_storage.d_values, max_num_elements * sizeof(V)));
		CUDA_SAFE_CALL( cudaMalloc( (void**) &device_storage.d_alt_values, max_num_elements * sizeof(V)));
	}
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_storage.d_spine, max_num_spine_elements * sizeof(unsigned int)) );
	CUDA_SAFE_CALL( cudaMalloc( (void**) &device_storage.d_from_alt_storage, 2 * sizeof(bool)) );

	// Run combinations of specified problem-sizes & max-grid-sizes
	for (int i = 0; i < num_elements_list_size; i++) {

		if (num_elements_list[i] > max_num_elements) {
			printf("Problem size %d too large\n", num_elements_list[i]);
			continue;
		}

		// Regenerate random keys if specified
		if ((i > 0) && g_regen) {
			for (unsigned int j = 0; j < num_elements_list[i]; j++) {
				RandomBits<K>(h_keys[j], g_entropy_reduction);
			}
		}
		
		for (int j = 0; j < num_max_grid_sizes; j++) {

			// Run a dummy kernel to demarcate the start of this set of iterations in the counter logs
			DummyKernel<<<1,1,0>>>();

			// Run the timing test 
			TimedSort<K, V, KEYS_ONLY>(num_elements_list[i], max_grid_sizes[j], h_keys, h_keys_result, device_storage, iterations);
		}
	}
	
    // cleanup memory
	free(h_keys);
	CUDA_SAFE_CALL(cudaFree(device_storage.d_keys));
	CUDA_SAFE_CALL(cudaFree(device_storage.d_alt_keys));
	CUDA_SAFE_CALL(cudaFree(device_storage.d_spine));
	CUDA_SAFE_CALL(cudaFree(device_storage.d_from_alt_storage));
	if (!KEYS_ONLY) {
		CUDA_SAFE_CALL(cudaFree(device_storage.d_values));
		CUDA_SAFE_CALL(cudaFree(device_storage.d_alt_values));
	}	
}


template<typename K>
void TestSort(
	int value_bytes,
	unsigned int iterations,
	int* num_elements_list,
	unsigned int num_elements_list_size,
	int* max_grid_sizes,
	unsigned int num_max_grid_sizes)
{
	switch (value_bytes) {
	case 0:		// keys only
		TestSort<K, KeysOnlyType, true>(iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	case 4:		// 32-bit values
		TestSort<K, unsigned int, false>(iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	case 8:		// 64-bit values
		TestSort<K, unsigned long long, false>(iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	case 16:	// 128-bit values
		TestSort<K, uint4, false>(iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	default: 
		fprintf(stderr, "Invalid payload size.  Exiting.\n");
	}
}



/******************************************************************************
 * Main
 ******************************************************************************/

int main( int argc, char** argv) {

	CUT_DEVICE_INIT(argc, argv);

	// presently deterministic
	// srand(time(NULL));	
	srand(0);				

	unsigned int num_elements 				= 512;
    int max_grid_size 						= -1;	// let API determine best grid size
	unsigned int iterations  				= 1;
	char *num_elements_list_filename 		= NULL;
	char *max_grid_sizes_filename 			= NULL;
	int key_bytes							= 4;
	int value_bytes							= 0;
	int* num_elements_list 					= NULL;
	int* max_grid_sizes 					= NULL;
	unsigned int num_elements_list_size;
	unsigned int num_max_grid_sizes;

    //
	// Check command line arguments
    //

    if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 0;
	}

    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", (int*)&iterations);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "key-bytes", (int*)&key_bytes);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "value-bytes", (int*)&value_bytes);
	cutGetCmdLineArgumentstr( argc, (const char**) argv, "n-input", &num_elements_list_filename);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "max-blocks", (int*)&max_grid_size);
	cutGetCmdLineArgumentstr( argc, (const char**) argv, "max-blocks-input", &max_grid_sizes_filename);
	cutGetCmdLineArgumenti( argc, (const char**) argv, "entropy-reduction", (int*)&g_entropy_reduction);
	if (g_verbose2 = cutCheckCmdLineFlag( argc, (const char**) argv, "v2")) {
		g_verbose = true;
	} else {
		g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");
	}
	g_verify = !cutCheckCmdLineFlag( argc, (const char**) argv, "noverify");
	g_regen = cutCheckCmdLineFlag( argc, (const char**) argv, "regen");
	
	// Attempt to read list of problem sizes to run
	ReadList(
		num_elements_list, 
		num_elements_list_size, 
		num_elements_list_filename, 
		num_elements); 

	// Attempt to read list of max-grid-sizes to run
	ReadList(
		max_grid_sizes, 
		num_max_grid_sizes, 
		max_grid_sizes_filename, 
		max_grid_size); 
	
	// Execute test(s)

	switch (key_bytes) {
	case 1:		// 8-bit keys
		TestSort<unsigned char>(value_bytes, iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	case 2:		// 16-bit keys
		TestSort<unsigned short>(value_bytes, iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	case 4:		// 32-bit keys
		TestSort<unsigned int>(value_bytes, iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	case 8:		// 64-bit keys
		TestSort<unsigned long long>(value_bytes, iterations, num_elements_list, num_elements_list_size, max_grid_sizes, num_max_grid_sizes);
		break;
	default: 
		fprintf(stderr, "Invalid key size.  Exiting.\n");
	}
}



