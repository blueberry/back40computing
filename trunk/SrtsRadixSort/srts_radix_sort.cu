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
// Radix Sorting API
//
// The interface for radix sorting is LaunchSort() below.  
//------------------------------------------------------------------------------

#ifndef _SRTS_RADIX_SORT_DRIVER_H_
#define _SRTS_RADIX_SORT_DRIVER_H_

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>


//------------------------------------------------------------------------------
// Sorting includes
//------------------------------------------------------------------------------

// Kernel includes
#include <kernel/srts_reduction_kernel.cu>
#include <kernel/srts_spine_kernel.cu>
#include <kernel/srts_scanscatter_kernel.cu>


//------------------------------------------------------------------------------
// Debugging options
//------------------------------------------------------------------------------

bool SRTS_DEBUG = false;


//------------------------------------------------------------------------------
// Routines
//------------------------------------------------------------------------------


/**
 * Heuristic for determining the number of CTAs to launch.
 *   
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in] 		max_grid_size  
 * 		Maximum allowable number of CTAs to launch.  A value of -1 indicates 
 * 		that the default value should be used.
 * 
 * @return The actual number of CTAs that should be launched
 */
unsigned int GridSize(
	unsigned int num_elements, 
	int max_grid_size,
	unsigned int cycle_elements,
	cudaDeviceProp device_props,
	unsigned int sm_version, 
	bool keys_only) 
{
	const unsigned int SINGLE_CTA_CUTOFF = 0;		// right now zero; we have no single-cta sorting

	// find maximum number of threadblocks if "use-default"
	if (max_grid_size == -1) {

		if (num_elements <= SINGLE_CTA_CUTOFF) {

			// The problem size is too small to warrant a two-level reduction: 
			// use only one stream-processor
			max_grid_size = 1;

		} else {

			if (sm_version <= 120) {
				
				// G80/G90
				max_grid_size = device_props.multiProcessorCount * 4;
				
			} else if (sm_version < 200) {
				
				// GT200 (has some kind of TLB or icache drama)
				unsigned int orig_max_grid_size = device_props.multiProcessorCount * SRTS_BULK_CTA_OCCUPANCY(sm_version);
				if (keys_only) { 
					orig_max_grid_size *= (num_elements + (1024 * 1024 * 96) - 1) / (1024 * 1024 * 96);
				} else {
					orig_max_grid_size *= (num_elements + (1024 * 1024 * 64) - 1) / (1024 * 1024 * 64);
				}
				max_grid_size = orig_max_grid_size;

				if (num_elements / cycle_elements > max_grid_size) {
	
					double multiplier1 = 4.0;
					double multiplier2 = 16.0;

					double delta1 = 0.068;
					double delta2 = 0.127;	
	
					unsigned int dividend = (num_elements + cycle_elements - 1) / cycle_elements;
	
					while(true) {
	
						double quotient = ((double) dividend) / (multiplier1 * max_grid_size);
						quotient -= (int) quotient;
						if ((quotient > delta1) && (quotient < 1 - delta1)) {

							quotient = ((double) dividend) / (multiplier2 * max_grid_size / 3.0);
							quotient -= (int) quotient;
							if ((quotient > delta2) && (quotient < 1 - delta2)) {
								break;
							}
						}
						
						if (max_grid_size == orig_max_grid_size - 2) {
							max_grid_size = orig_max_grid_size - 30;
						} else {
							max_grid_size -= 1;
						}
					}
				}
			} else {
				
				// GF100
				max_grid_size = 418;
			}
		}
	}

	// Calculate the actual number of threadblocks to launch.  Initially
	// assume that each threadblock will do only one cycle_elements worth 
	// of work, but then clamp it by the "max" restriction derived above
	// in order to accomodate the "single-sp" and "saturated" cases.

	unsigned int grid_size = num_elements / cycle_elements;
	if (grid_size == 0) {
		grid_size = 1;
	}
	if (grid_size > max_grid_size) {
		grid_size = max_grid_size;
	} 

	return grid_size;
}


/**
 * 
 */
template <typename K, typename V, bool KEYS_ONLY, unsigned int RADIX_BITS, unsigned int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
cudaError_t SortDigit(
	cudaDeviceProp device_props,
	unsigned int sm_version,
	unsigned int num_elements,
	unsigned int grid_size,
	const GlobalStorage<K, V> &problem_storage,
	const CtaDecomposition &work_decomposition,
	unsigned int spine_block_elements) 
{
	unsigned int threads = SRTS_THREADS;
	unsigned int dynamic_smem;

	//-------------------------------------------------------------------------
	// Counting Reduction
	//-------------------------------------------------------------------------

	// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
	if ((sm_version == 130) && (num_elements > device_props.multiProcessorCount * 2 * SRTS_CYCLE_ELEMENTS(sm_version, K, V))) 
			FlushKernel<<<grid_size, SRTS_THREADS, 3000>>>();

	// Fermi gets the same smem allocation for every kernel launch
	dynamic_smem = (sm_version >= 200) ? 5448 - 2048 : 0;
	RakingReduction<K, V, RADIX_BITS, BIT, PreprocessFunctor> <<<grid_size, threads, dynamic_smem>>>(
			problem_storage.keys,
			problem_storage.temp_spine,
			work_decomposition);

	
	//-------------------------------------------------------------------------
	// Spine
	//-------------------------------------------------------------------------
	
	// Fermi gets the same smem allocation for every kernel launch
	dynamic_smem = (sm_version >= 200) ? 5448 - 784 : 0;
	SrtsScanSpine<<<grid_size, SRTS_SPINE_THREADS, dynamic_smem>>>(
		problem_storage.temp_spine,
		problem_storage.temp_spine,
		spine_block_elements);

	
	//-------------------------------------------------------------------------
	// Scanning Scatter
	//-------------------------------------------------------------------------
	
	// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
	if ((sm_version == 130) && (num_elements > device_props.multiProcessorCount * 2 * SRTS_CYCLE_ELEMENTS(sm_version, K, V))) 
			FlushKernel<<<grid_size, SRTS_THREADS, 3000>>>();

	dynamic_smem = 0;
	SrtsScanDigitBulk<K, V, KEYS_ONLY, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor> <<<grid_size, threads, 0>>>(
		problem_storage.temp_spine,
		problem_storage.keys,
		problem_storage.temp_keys,
		problem_storage.data,
		problem_storage.temp_data,
		work_decomposition);
	
	return cudaSuccess;
}


/**
 * Enactor template to specialize sorting depending upon key size.  
 * 
 * N.B.: Again, we follow this meta-programming pattern of partial template 
 * specialization for structures.  Alternatively, we could use a single 
 * procedure to orchestrate all of the digit passes for any unsigned integral 
 * key type: code up passes for the largest key (eight bytes) and simply 
 * predicate each digit-place pass upon the size of the key. Doing so, however, 
 * results in two bad things: 
 *   (1) NVCC generates kernels before dead-code-elimination elides those call 
 *       sites, resulting in unnecesary kernels for digit places on keys that 
 *       don't exist (e.g., digit place for bits 52-55 on a unsigned int). 
 *       Code bloat.
 *   (2) As a consequence, you get a mess of shift-bounds warnings for these 
 *       illegitimate kernels that can never be called. 
 */
template <typename K, typename V, bool KEYS_ONLY,unsigned int RADIX_BITS, typename PreprocessFunctor, typename PostprocessFunctor> struct SortingEnactor;


/**
 * Enactor for sorting 1-byte keys 
 */
template <typename V, bool KEYS_ONLY,typename PreprocessFunctor, typename PostprocessFunctor> 
struct SortingEnactor<unsigned char, V, KEYS_ONLY, 4, PreprocessFunctor, PostprocessFunctor> {
	
	static cudaError_t EnactDigitPlacePasses(
			cudaDeviceProp device_props,
			unsigned int sm_version,
			unsigned int num_elements,
			unsigned int spine_block_elements,
			unsigned int grid_size,
			GlobalStorage<unsigned char, V> &problem_storage,
			GlobalStorage<unsigned char, V> &swizzle_storage,
			const CtaDecomposition &work_decomposition)
	{
		typedef unsigned char ConvertedKeyType;
		
		// Sort using 4-bit radix digit passes  
		
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 0, PreprocessFunctor, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements);
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 4, NopFunctor<ConvertedKeyType>, PostprocessFunctor> (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 

		return cudaSuccess;
	}
};


/**
 * Enactor for sorting 2-byte keys 
 */
template <typename V, bool KEYS_ONLY,typename PreprocessFunctor, typename PostprocessFunctor> 
struct SortingEnactor<unsigned short, V, KEYS_ONLY, 4, PreprocessFunctor, PostprocessFunctor> {
	
	static cudaError_t EnactDigitPlacePasses(
			cudaDeviceProp device_props,
			unsigned int sm_version,
			unsigned int num_elements,
			unsigned int spine_block_elements,
			unsigned int grid_size,
			GlobalStorage<unsigned short, V> &problem_storage,
			GlobalStorage<unsigned short, V> &swizzle_storage,
			const CtaDecomposition &work_decomposition)
	{
		typedef unsigned short ConvertedKeyType;

		// Sort using 4-bit radix digit passes  
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 0,  PreprocessFunctor,            NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements);
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 12, NopFunctor<ConvertedKeyType>, PostprocessFunctor>            (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 

		return cudaSuccess;
	}
};


/**
 * Enactor for sorting 4-byte keys 
 */
template <typename V, bool KEYS_ONLY,typename PreprocessFunctor, typename PostprocessFunctor> 
struct SortingEnactor<unsigned int, V, KEYS_ONLY, 4, PreprocessFunctor, PostprocessFunctor> {
	
	static cudaError_t EnactDigitPlacePasses(
			cudaDeviceProp device_props,
			unsigned int sm_version,
			unsigned int num_elements,
			unsigned int spine_block_elements,
			unsigned int grid_size,
			GlobalStorage<unsigned int, V> &problem_storage,
			GlobalStorage<unsigned int, V> &swizzle_storage,
			const CtaDecomposition &work_decomposition)
	{
		typedef unsigned int ConvertedKeyType;
		
		// Sort using 4-bit radix digit passes  
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 0,  PreprocessFunctor,            NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements);
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 28, NopFunctor<ConvertedKeyType>, PostprocessFunctor>            (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 

		return cudaSuccess;
	}
};


/**
 * Enactor for sorting 8-byte keys 
 */
template <typename V, bool KEYS_ONLY,typename PreprocessFunctor, typename PostprocessFunctor> 
struct SortingEnactor<unsigned long long, V, KEYS_ONLY, 4, PreprocessFunctor, PostprocessFunctor> {
	
	static cudaError_t EnactDigitPlacePasses(
			cudaDeviceProp device_props,
			unsigned int sm_version,
			unsigned int num_elements,
			unsigned int spine_block_elements,
			unsigned int grid_size,
			GlobalStorage<unsigned long long, V> &problem_storage,
			GlobalStorage<unsigned long long, V> &swizzle_storage,
			const CtaDecomposition &work_decomposition)
	{
		typedef unsigned long long ConvertedKeyType;
		
		// Sort using 4-bit radix digit passes  
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 0,  PreprocessFunctor,            NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements);
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 28, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 32, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 36, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 40, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 44, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 48, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 52, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 56, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> > (device_props, sm_version, num_elements, grid_size, problem_storage, work_decomposition, spine_block_elements); 
		SortDigit<ConvertedKeyType, V, KEYS_ONLY, 4, 60, NopFunctor<ConvertedKeyType>, PostprocessFunctor>            (device_props, sm_version, num_elements, grid_size, swizzle_storage, work_decomposition, spine_block_elements); 

		return cudaSuccess;
	}
};



/**
 */
template <typename K, typename V, bool KEYS_ONLY>
cudaError_t EnactSort(
	unsigned int num_elements, 
	GlobalStorage<K, V> &problem_storage,	
	int max_grid_size) 
{
	// Sort using 4-bit radix digit passes  
	const unsigned int RADIX_BITS = 4;
	
	//
	// Get device properties
	// 
	
	int current_device;
	cudaDeviceProp device_props;
	cudaGetDevice(&current_device);
	cudaGetDeviceProperties(&device_props, current_device);
	unsigned int sm_version = device_props.major * 100 + device_props.minor * 10;
	
	//
	// Determine number of CTAs to launch, shared memory, cycle elements, etc.
	//

	unsigned int cycle_elements = SRTS_CYCLE_ELEMENTS(sm_version, K, V);
	unsigned int grid_size = GridSize(num_elements, max_grid_size, cycle_elements, device_props, sm_version, KEYS_ONLY);
	
	//
	// Determine how many elements each CTA will process
	//
	
	unsigned int total_cycles 			= num_elements / cycle_elements;
	unsigned int cycles_per_block 		= total_cycles / grid_size;						
	unsigned int extra_cycles 			= total_cycles - (cycles_per_block * grid_size);
	unsigned int spine_cycles 			= ((grid_size * (1 << RADIX_BITS)) + SRTS_SPINE_CYCLE_ELEMENTS - 1) / SRTS_SPINE_CYCLE_ELEMENTS;
	unsigned int spine_block_elements 	= spine_cycles * SRTS_SPINE_CYCLE_ELEMENTS;

	CtaDecomposition work_decomposition = {
		extra_cycles,										// num_big_blocks
		(cycles_per_block + 1) * cycle_elements,			// big_block_elements
		cycles_per_block * cycle_elements,					// normal_block_elements
		num_elements - (total_cycles * cycle_elements)};	// extra_elements_last_block

	//
	// Allocate device memory for temporary storage (if necessary)
	//

	if (problem_storage.temp_keys == NULL) {
		cudaMalloc((void**) &problem_storage.temp_keys, num_elements * sizeof(K));
	}
	if (!KEYS_ONLY && (problem_storage.temp_data == NULL)) {
		cudaMalloc((void**) &problem_storage.temp_data, num_elements * sizeof(V));
	}
	if (problem_storage.temp_spine == NULL) {
		cudaMalloc((void**) &problem_storage.temp_spine, spine_block_elements * sizeof(unsigned int));
	}
	
	//
	// Create storage management structures
	//
	
	// Determine suitable type of unsigned byte storage to use for keys 
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;
	
	// Copy storage pointers to an appropriately typed stucture 
	GlobalStorage<ConvertedKeyType, V> converted_storage = {
		(ConvertedKeyType *) problem_storage.keys,				// keys
		problem_storage.data,									// data
		(ConvertedKeyType *) problem_storage.temp_keys,			// temp_keys
		problem_storage.temp_data,								// temp_data
		problem_storage.temp_spine};							// temp_spine

	// Create a secondary structure for flip-flopping between passes
	GlobalStorage<ConvertedKeyType, V> swizzle_storage = {
		(ConvertedKeyType *) problem_storage.temp_keys,			// keys
		problem_storage.temp_data,								// data
		(ConvertedKeyType *) problem_storage.keys,				// temp_keys
		problem_storage.data,									// temp_data
		problem_storage.temp_spine};							// temp_spine

	// 
	// Enact the sorting operation
	//
	
	if (SRTS_DEBUG) {
		printf("RakingReduction <<<%d,%d>>>(\n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d)\n\n",
			grid_size, SRTS_THREADS, 0, SRTS_CYCLE_ELEMENTS(sm_version, K, V), work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
		printf("SrtsScanSpine<<<%d,%d>>>(\n\tspine_block_elements: %d)\n\n", 
			grid_size, SRTS_SPINE_THREADS, spine_block_elements);
		printf("SrtsScanDigitBulk <<<%d,%d>>>(\n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d,\n\textra_elements_last_block: %d)\n\n", 
			grid_size, SRTS_THREADS, SRTS_CYCLE_ELEMENTS(sm_version, K, V), work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
	}	

	cudaError_t retval = SortingEnactor<ConvertedKeyType, V, KEYS_ONLY, RADIX_BITS, PreprocessKeyFunctor<K>, PostprocessKeyFunctor<K> >::EnactDigitPlacePasses(
		device_props,
		sm_version,
		num_elements,
		spine_block_elements,
		grid_size,
		converted_storage,
		swizzle_storage,
		work_decomposition);
	
	// 
	// Copy (possibly updated/swizzled) storage pointers back
	//

	problem_storage.keys = (K*) converted_storage.keys;
	problem_storage.data = (V*) converted_storage.data;
	problem_storage.temp_keys = (K*) converted_storage.temp_keys;
	problem_storage.temp_data = (V*) converted_storage.temp_data;
	problem_storage.temp_spine = converted_storage.temp_spine;
	
	return retval;
}



/**
 * Launches a simple, key-value sort.
 * 
 * IMPORTANT NOTES: The device storage backing the specified input vectors of 
 * keys (and data) will be modified.  (I.e., treat this as an in-place sort.)  
 * 
 * Additionally, the pointers in the problem_storage structure may be updated 
 * (a) depending upon the number of digit-place sorting passes needed, and (b) 
 * whether or not the caller has already allocated temporary storage.  
 * 
 * The sorted results will always be referenced by problem_storage.keys (and 
 * problem_storage.data).  However, for an odd number of sorting passes (uncommon)
 * these results will actually be backed by the storage initially allocated for 
 * by problem_storage.temp_keys (and problem_storage.temp_data).  If so, 
 * problem_storage.temp_keys and problem_storage.temp_keys will be updated to 
 * reference the original problem_storage.keys and problem_storage.data in order 
 * to facilitate cleanup.  
 * 
 * This means it is important to avoid keeping stale copies of device pointers 
 * to keys/data; you will want to re-reference the pointers in problem_storage.
 * 
 * template-param K
 * 		Type of keys to be sorted
 *
 * template-param V
 * 		Type of values to be sorted.
 *
 * @param[in] 		num_elements 
 * 		Length (in elements) of the vector to sort
 *
 * @param[in/out] 	problem_storage 
 * 		Device vectors of keys and values to sort, and ancillary storage 
 * 		needed by the sorting kernels. See the IMPORTANT NOTES above. 
 * 
 * 		The problem_storage.[temp_keys|temp_data|temp_spine] fields are 
 * 		temporary storage needed by the sorting kernels.  To facilitate 
 * 		speed, callers are welcome to re-use this storage for same-sized 
 * 		(or smaller) sortign problems. If NULL, these storage vectors will be 
 *      allocated by this routine (and must be subsequently cuda-freed by 
 *      the caller).
 *
 * @param[in] 		max_grid_size  
 * 		Maximum allowable number of CTAs to launch.  The default value of -1 indicates 
 * 		that the dispatch logic should select an appropriate value for the target device.
 * 
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <typename K, typename V>
cudaError_t LaunchKeyValueSort(
	unsigned int num_elements, 
	GlobalStorage<K, V> &problem_storage,	
	int max_grid_size = -1) 
{
	return EnactSort<K, V, false>(
		num_elements, 
		problem_storage,	
		max_grid_size);
}



/**
 * Launches a simple, keys-only sort. 
 * 
 * IMPORTANT NOTES: The device storage backing the specified input vectors of 
 * keys (and data) will be modified.  (I.e., treat this as an in-place sort.)  
 * 
 * Additionally, the pointers in the problem_storage structure may be updated 
 * (a) depending upon the number of digit-place sorting passes needed, and (b) 
 * whether or not the caller has already allocated temporary storage.  
 * 
 * The sorted results will always be referenced by problem_storage.keys (and 
 * problem_storage.data).  However, for an odd number of sorting passes (uncommon)
 * these results will actually be backed by the storage initially allocated for 
 * by problem_storage.temp_keys (and problem_storage.temp_data).  If so, 
 * problem_storage.temp_keys and problem_storage.temp_keys will be updated to 
 * reference the original problem_storage.keys and problem_storage.data in order 
 * to facilitate cleanup.  
 * 
 * This means it is important to avoid keeping stale copies of device pointers 
 * to keys/data; you will want to re-reference the pointers in problem_storage.
 * 
 * template-param K
 * 		Type of keys to be sorted
 *
 * @param[in] 		num_elements 
 * 		Length (in elements) of the vector to sort
 *
 * @param[in/out] 	problem_storage 
 * 		Device vectors of keys (and values) to sort, and ancillary storage 
 * 		needed by the sorting kernels. See the IMPORTANT NOTES above. 
 * 
 * 		The problem_storage.data and problem_storage.temp_data fields may 
 * 		be both NULL (if so, keys-only sorting will be performed.)  
 * 
 * 		The problem_storage.[temp_keys|temp_data|temp_spine] fields are 
 * 		temporary storage needed by the sorting kernels.  To facilitate 
 * 		speed, callers are welcome to re-use this storage for same-sized 
 * 		(or smaller) sortign problems. If NULL, these storage vectors will be 
 *      allocated by this routine (and must be subsequently cuda-freed by 
 *      the caller).
 *
 * @param[in] 		max_grid_size  
 * 		Maximum allowable number of CTAs to launch.  The default value of -1 indicates 
 * 		that the dispatch logic should select an appropriate value for the target device.
 * 
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <typename K>
cudaError_t LaunchKeysOnlySort(
	unsigned int num_elements, 
	GlobalStorage<K> &problem_storage,	
	int max_grid_size = -1) 
{
	// Save off the satellite data pointer if for some reason the caller did not set it to NULL
	K *satellite_data = problem_storage.data;
	problem_storage.data = NULL;
	
	cudaError_t retval = EnactSort<K, K, true>(
		num_elements, 
		problem_storage,	
		max_grid_size);
	
	// Restore satellite data pointer
	problem_storage.data = satellite_data;
	
	return retval;
}





#endif

