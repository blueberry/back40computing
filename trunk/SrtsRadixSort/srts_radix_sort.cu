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
// Routines
//------------------------------------------------------------------------------

template <typename T>
void Swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


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
	unsigned int sm_version) 
{
	const unsigned int SINGLE_CTA_CUTOFF 		= 0;		// right now zero; we have no single-cta sorting

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
				
				// GT200
				max_grid_size = device_props.multiProcessorCount * SRTS_BULK_CTA_OCCUPANCY(sm_version);
				if (num_elements / cycle_elements > max_grid_size) {
	
					unsigned int multiplier = 16;
	
					double top_delta = 0.078;	
					double bottom_delta = 0.078;
	
					unsigned int dividend = (num_elements + cycle_elements - 1) / cycle_elements;
	
					while(true) {
	
						double quotient = ((double) dividend) / (multiplier * max_grid_size);
						quotient -= (int) quotient;
	
						if ((quotient > top_delta) && (quotient < 1 - bottom_delta)) {
							break;
						}
	
						if (max_grid_size == 147) {
							max_grid_size = 120;
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



template <typename K, typename V, unsigned int RADIX_BITS, unsigned int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
cudaError_t SortDigit(
	bool verbose,
	cudaDeviceProp device_props,
	unsigned int sm_version,
	unsigned int num_elements,
	unsigned int grid_size,
	unsigned int threads,
	const GlobalStorage<K, V> &problem_storage,
	const CtaDecomposition &work_decomposition,
	unsigned int spine_block_elements) 
{

	//-------------------------------------------------------------------------
	// Counting Reduction
	//-------------------------------------------------------------------------

	// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
	if ((sm_version == 130) && (num_elements > device_props.multiProcessorCount * 2 * SRTS_CYCLE_ELEMENTS(sm_version, K, V))) 
			FlushKernel<<<grid_size, SRTS_THREADS, 3000>>>();

	// Fermi gets the same smem allocation for every kernel launch
	unsigned int dynamic_smem = (sm_version >= 200) ? 5448 - 2048 : 0;
	if (verbose) {
		printf("RakingReduction <<<%d,%d,%d>>>(\n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d)\n\n",
			grid_size, threads, dynamic_smem, SRTS_CYCLE_ELEMENTS(sm_version, K, V), work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
	}
	RakingReduction<K, V, RADIX_BITS, BIT, PreprocessFunctor> <<<grid_size, threads, dynamic_smem>>>(
			problem_storage.keys,
			problem_storage.temp_spine,
			work_decomposition);

	
	//-------------------------------------------------------------------------
	// Spine
	//-------------------------------------------------------------------------
	
	// Fermi gets the same smem allocation for every kernel launch
	dynamic_smem = (sm_version >= 200) ? 5448 - 784 : 0;
	if (verbose) {
		printf("SrtsScanSpine<<<%d,%d,%d>>>(\n\tspine_block_elements: %d)\n\n", 
			grid_size, SRTS_SPINE_THREADS, dynamic_smem,
			spine_block_elements);
	}
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
	if (verbose) {
		printf("SrtsScanDigitBulk <<<%d,%d,%d>>>(\n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d,\n\textra_elements_last_block: %d)\n\n", 
			grid_size, threads, dynamic_smem, SRTS_CYCLE_ELEMENTS(sm_version, K, V), work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
	}
	
	if (problem_storage.data == NULL) {
		// keys-only
		SrtsScanDigitBulk<K, V, true, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor> <<<grid_size, threads, 0>>>(
			problem_storage.temp_spine,
			problem_storage.keys,
			problem_storage.temp_keys,
			problem_storage.data,
			problem_storage.temp_data,
			work_decomposition);
	} else {
		// key-value
		SrtsScanDigitBulk<K, V, false, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor> <<<grid_size, threads, 0>>>(
			problem_storage.temp_spine,
			problem_storage.keys,
			problem_storage.temp_keys,
			problem_storage.data,
			problem_storage.temp_data,
			work_decomposition);
	}
	
	return cudaSuccess;
}








/**
 * Launches a simple, two-level sort. Can sort keys-only and key-value pairs.
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
 * 		Type of values to be sorted.  If you are doing keys-only sorting (i.e., 
 * 		problem_storage.data == NULL), kindly specify the same key type (type K)
 * 		as the value type V as well.
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
 * @param[in] 		verbose  
 * 		Flag whether or not to print launch information to stdout
 *
 * @param[in] 		max_grid_size  
 * 		Maximum allowable number of CTAs to launch.  The default value of -1 indicates 
 * 		that the dispatch logic should select an appropriate value for the target device.
 * 
 * @return cudaSuccess on success, error enumeration otherwise
 */
template <typename K, typename V>
cudaError_t LaunchSort(
	unsigned int num_elements, 
	GlobalStorage<K, V> &problem_storage,	
	bool verbose,
	int max_grid_size = -1) 
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

	unsigned int threads = SRTS_THREADS; 
	unsigned int cycle_elements = SRTS_CYCLE_ELEMENTS(sm_version, K, V);
	unsigned int grid_size = GridSize(num_elements, max_grid_size, cycle_elements, device_props, sm_version);

	
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
	if ((problem_storage.data != NULL) && (problem_storage.temp_data == NULL)) {
		cudaMalloc((void**) &problem_storage.temp_data, num_elements * sizeof(V));
	}
	if (problem_storage.temp_spine == NULL) {
		cudaMalloc((void**) &problem_storage.temp_spine, spine_block_elements * sizeof(unsigned int));
	}
	

	//
	// Cast keys to unsigned type suitable for radix sorting and create structures
	// for flip-flopping between passes
	//

	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;
	
	GlobalStorage<ConvertedKeyType, V> flip_storage = {
		(ConvertedKeyType *) problem_storage.keys,				// keys
		problem_storage.data,									// data
		(ConvertedKeyType *) problem_storage.temp_keys,			// temp_keys
		problem_storage.temp_data,								// temp_data
		problem_storage.temp_spine};							// temp_spine

	GlobalStorage<ConvertedKeyType, V> flop_storage = {
		(ConvertedKeyType *) problem_storage.temp_keys,			// keys
		problem_storage.temp_data,								// data
		(ConvertedKeyType *) problem_storage.keys,				// temp_keys
		problem_storage.data,									// temp_data
		problem_storage.temp_spine};							// temp_spine

	
	//
	// Sort using RADIX_BITS-bit radix digit passes  
	//

	bool swizzle_ptrs = false;
	
	if (RADIX_BITS == 4) {
		
		SortDigit<ConvertedKeyType, V, RADIX_BITS, 0,  PreprocessKeyFunctor<K>, NopFunctor<ConvertedKeyType> > 		(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements);
		verbose = false;
		if (sizeof(K) > 1) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		if (sizeof(K) > 2) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		if (sizeof(K) > 3) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		if (sizeof(K) > 4) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 28, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 32, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		if (sizeof(K) > 5) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 36, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 40, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		if (sizeof(K) > 6) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 44, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 48, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		if (sizeof(K) > 7) {
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 52, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 
			SortDigit<ConvertedKeyType, V, RADIX_BITS, 56, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >	(verbose, device_props, sm_version, num_elements, grid_size, threads, flip_storage, work_decomposition, spine_block_elements); 
		}
		SortDigit<ConvertedKeyType, V, RADIX_BITS, (sizeof(K) * 8) - RADIX_BITS, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >		(verbose, device_props, sm_version, num_elements, grid_size, threads, flop_storage, work_decomposition, spine_block_elements); 

		// The output is back in d_in_keys.  No pointer-swizzling between problem_storage.keys and problem_storage.temp_keys needed.
		swizzle_ptrs = false;

	} else {
		
		fprintf(stderr, "Todo: make kernel calls for other numbers of radix digit numerals.\n");
		return cudaErrorNotYetImplemented;
	}
	
	// Pointer swizzle if necessary
	if (swizzle_ptrs) {
		Swap<K*>(problem_storage.keys, problem_storage.temp_keys);
		Swap<V*>(problem_storage.data, problem_storage.temp_data);
	}
	
	return cudaSuccess;
}



#endif

