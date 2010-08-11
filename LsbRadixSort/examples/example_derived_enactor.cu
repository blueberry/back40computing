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

#include <radixsort_api.cu>			// Sorting includes
#include <test_utils.cu>			// Utilities and correctness-checking
#include <cutil.h>					// Utilities for commandline parsing

using namespace b40c;


/******************************************************************************
 * Defines, constants, globals 
 ******************************************************************************/

bool g_verbose;



/******************************************************************************
 * Customized 17-bit in-place enactor
 ******************************************************************************/


/**
 * Example 5-pass custom sorting enactor that is specialized for 17-effective-bit 
 * uint32 key types.  
 * 
 * It also demonstrates the use of a third storage array to perform in-place sorting 
 * in the face of odd-passes without having to resort to (a) re-aliasing device 
 * input pointers, or (b) an extra memcpy back to the original input array.
 */
template <typename V = KeysOnlyType>
class AmberRadixSortingEnactor : public BaseRadixSortingEnactor<unsigned int, V>
{
protected:

	typedef BaseRadixSortingEnactor<unsigned int, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;
	
	unsigned int *d_inner_keys;
	V *d_inner_values;

	cudaError_t EnactDigitPlacePasses(const RadixSortStorage<ConvertedKeyType, V> &converted_storage)
	{
		
		//
		// Allocate a second set of temporary vectors to be used for flip-flopping between digit_place 
		// passes in the "inner" passes
		//
		
		if (!d_inner_keys) cudaMalloc((void**) &d_inner_keys, Base::_num_elements * sizeof(unsigned int));							// Alloc the third keys vector
		if (!IsKeysOnly<V>() && !d_inner_values) cudaMalloc((void**) &d_inner_values, Base::_num_elements * sizeof(V));				// Same for values...

		//
		// Create a second storage management structure that will contains our pair of temporary vectors
		// to be used for flip-flopping between digit_place passes in the "inner" passes
		//
		
		RadixSortStorage<ConvertedKeyType, V> inner_pass_storage = converted_storage;
		inner_pass_storage.d_keys = d_inner_keys;
		inner_pass_storage.d_values = d_inner_values;
		
	
		//
		// Create a third storage management structure that will convey inner_pass_storage.d_keys to 
		// converted_storage.d_keys on the final pass.
		//
		
		RadixSortStorage<ConvertedKeyType, V> final_pass_storage = inner_pass_storage;
		final_pass_storage.d_alt_keys = converted_storage.d_keys;
		if (!IsKeysOnly<V>()) final_pass_storage.d_alt_values = converted_storage.d_values;
		
		//
		// Enact the sorting procedure, making sure that the first and last passes are performed 
		// regardless of digit-uniformity in order to guarantee that the sorted output
		// always ends up back the the orignial input array.
		//
		
		Base::template DigitPlacePass<0, 4, 0,  MandatoryPassNopFunctor<ConvertedKeyType>, MandatoryPassNopFunctor<ConvertedKeyType> >(converted_storage);		// d_keys (orig) -> d_alt_keys (orig, inner)
		Base::template DigitPlacePass<1, 4, 4,  MandatoryPassNopFunctor<ConvertedKeyType>, MandatoryPassNopFunctor<ConvertedKeyType> >(inner_pass_storage); 	// d_alt_keys (orig, inner) -> d_keys (inner)
		Base::template DigitPlacePass<2, 4, 8,  MandatoryPassNopFunctor<ConvertedKeyType>, MandatoryPassNopFunctor<ConvertedKeyType> >(inner_pass_storage); 	// d_keys (inner) -> d_alt_keys  (orig, inner)
		Base::template DigitPlacePass<3, 3, 12, MandatoryPassNopFunctor<ConvertedKeyType>, MandatoryPassNopFunctor<ConvertedKeyType> >(inner_pass_storage); 	// d_alt_keys (orig, inner) -> d_keys (inner)
		Base::template DigitPlacePass<4, 2, 15, MandatoryPassNopFunctor<ConvertedKeyType>, MandatoryPassNopFunctor<ConvertedKeyType> >(final_pass_storage); 	// d_keys (inner) -> d_keys (orig)

		return cudaSuccess;
	}

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		num_elements 
	 * 		Length (in elements) of the input to a sorting operation
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of -1 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	AmberRadixSortingEnactor(unsigned int num_elements, int max_grid_size = -1) : 
		Base::BaseRadixSortingEnactor(5, 4, num_elements, max_grid_size, false), d_inner_keys(NULL), d_inner_values(NULL) {}


	// Clean up inner temporary storage native to this enactor 
	cudaError_t CleanupTempStorage() 
	{
		if (d_inner_keys) cudaFree(d_inner_keys);
		if (d_inner_values) cudaFree(d_inner_values);
		
		return cudaSuccess;
	}
};




/******************************************************************************
 * Routines
 ******************************************************************************/

/**
 * Displays the commandline usage for this tool
 */
void Usage() 
{
	printf("\nderived_enactor [--device=<device index>] [--v] [--i=<num-iterations>] [--n=<num-elements>] [--keys-only]\n"); 
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
void CustomTimedSort(
	unsigned int num_elements, 
	unsigned int *h_keys,
	unsigned int iterations)
{
	printf("Custom enactor on 17-bit-effective keys, %d iterations, %d elements", iterations, num_elements);
	
	//
	// Allocate device storage and create sorting enactor  
	//

	RadixSortStorage<unsigned int> device_storage;	
	CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.d_keys, sizeof(unsigned int) * num_elements) );

	AmberRadixSortingEnactor<> sorting_enactor(num_elements);


	
	//
	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	//
	
	CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice) );		// copy keys
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
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice) );	// copy keys

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
    
    sorting_enactor.CleanupTempStorage();						// clean up internal enactor storage
    device_storage.CleanupTempStorage();						// clean up resort-allocated storage 

    CUDA_SAFE_CALL( cudaMemcpy(h_keys, device_storage.d_keys, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree(device_storage.d_keys) );		// clean up keys

    // Clean up events
	CUDA_SAFE_CALL( cudaEventDestroy(start_event) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop_event) );
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
void DefaultTimedSort(
	unsigned int num_elements, 
	unsigned int *h_keys,
	unsigned int iterations)
{
	printf("Default enactor on 17-bit-effective keys, %d iterations, %d elements", iterations, num_elements);
	
	//
	// Allocate device storage and create sorting enactor  
	//

	RadixSortStorage<unsigned int> device_storage;	
	CUDA_SAFE_CALL( cudaMalloc((void**) &device_storage.d_keys, sizeof(unsigned int) * num_elements) );

	RadixSortingEnactor<unsigned int> sorting_enactor(num_elements);


	
	//
	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	//
	
	CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice) );		// copy keys
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
		CUDA_SAFE_CALL( cudaMemcpy(device_storage.d_keys, h_keys, sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice) );	// copy keys

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
    
    device_storage.CleanupTempStorage();						// clean up resort-allocated storage 

    CUDA_SAFE_CALL( cudaMemcpy(h_keys, device_storage.d_keys, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaFree(device_storage.d_keys) );		// clean up keys

    // Clean up events
	CUDA_SAFE_CALL( cudaEventDestroy(start_event) );
	CUDA_SAFE_CALL( cudaEventDestroy(stop_event) );
}



/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of unsigned int elements, values of V elements, and then dispatches the problem 
 * to the GPU for the given number of iterations, displaying runtime information.
 *
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 */
void TestSort(
	unsigned int iterations,
	int num_elements, 
	bool custom)
{
    // Allocate the sorting problem on the host and fill the keys with random bytes

	unsigned int *h_keys = NULL;
	h_keys = (unsigned int*) malloc(num_elements * sizeof(unsigned int));

	// Use random bits
	for (unsigned int i = 0; i < num_elements; ++i) {
		RandomBits<unsigned int>(h_keys[i], 0);
		
		// only use 17 effective bits of key data
		h_keys[i] &= (1 << 17) - 1;
	}

    // Run the timing test
	if (custom) 
		CustomTimedSort(num_elements, h_keys, iterations);
	else 
		DefaultTimedSort(num_elements, h_keys, iterations);
    
	// Display sorted key data
	if (g_verbose) {
		printf("\n\nKeys:\n");
		for (int i = 0; i < num_elements; i++) {	
			PrintValue<unsigned int>(h_keys[i]);
			printf(", ");
		}
		printf("\n\n");
	}	
	
    // Verify solution
	VerifySort<unsigned int>(h_keys, num_elements, true);
	printf("\n");
	fflush(stdout);

	// Free our allocated host memory 
	if (h_keys != NULL) free(h_keys);
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

    //
	// Check command line arguments
    //

    if (cutCheckCmdLineFlag( argc, (const char**) argv, "help")) {
		Usage();
		return 0;
	}

    cutGetCmdLineArgumenti( argc, (const char**) argv, "i", (int*)&iterations);
    cutGetCmdLineArgumenti( argc, (const char**) argv, "n", (int*)&num_elements);
	g_verbose = cutCheckCmdLineFlag( argc, (const char**) argv, "v");

	TestSort(iterations, num_elements, true);	// custom 
	TestSort(iterations, num_elements, false); 	// default
}



