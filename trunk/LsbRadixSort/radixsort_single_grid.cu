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
 * Radix Sorting API
 *
 ******************************************************************************/

#pragma once

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 
#include <math.h> 
#include <float.h>

#include "b40c_error_synchronize.cu"
#include "kernel/radixsort_reduction_kernel.cu"
#include "kernel/radixsort_spine_kernel.cu"
#include "kernel/radixsort_scanscatter_kernel.cu"

#include "radixsort_multi_cta.cu"

namespace b40c {



/**
 * Storage management structure for device vectors.
 * 
 * The 0th elements of d_keys and d_values are used to point to the sorting
 * input.  After sorting, they will also point to the sorted output.
 * The remaining pointers are for temporary storage arrays needed
 * to stream data between sorting passes.  These arrays can be allocated
 * lazily upon first sorting by the sorting enactor, or a-priori by the 
 * caller.  (If user-allocated, they should be large enough to accomodate 
 * num_elements.)    
 * 
 * It is the caller's responsibility to free any non-NULL storage arrays when
 * no longer needed.  This storage can be re-used for subsequent sorting 
 * operations of the same size.   
 * 
 * NOTE: The enactor will only allocate storage for the third array for 
 * problems having an odd-number-of-passes; that way it can ensure the sorted 
 * results end up in d_keys[0]. 
 */
template <typename K, typename V = KeysOnlyType> 
struct SingleGridRadixSortStorage
{
	// Triple of device vector pointers for keys. 
	K* d_keys[3];
	
	// Triple of device vector pointers for values
	V* d_values[3];

	// Number of elements for sorting in the above vectors 
	int num_elements;
	
	// Constructor
	SingleGridRadixSortStorage(int num_elements) :
		num_elements(num_elements), 
		selector(0) 
	{
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_keys[2] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
		d_values[2] = NULL;
	}

	// Constructor
	SingleGridRadixSortStorage(int num_elements, K* keys, V* values = NULL) :
		num_elements(num_elements), 
	{
		d_keys[0] = keys;
		d_keys[1] = NULL;
		d_keys[2] = NULL;
		d_values[0] = values;
		d_values[1] = NULL;
		d_values[2] = NULL;
	}
};





/**
 * Base class for early-exit, multi-CTA radix sorting enactors.
 */
template <typename K, typename V = KeysOnlyType>
class SingleGridEnactor : public MultiCtaRadixSortingEnactor<K, V, SingleGridRadixSortStorage<K, V> >
{
private:
	
	// Typedef for base class
	typedef MultiCtaRadixSortingEnactor<K, V, SingleGridRadixSortStorage<K, V> > Base; 

protected:
	
	// Array of global synchronization counters, one for each threadblock
	int *d_sync;

	
public: 
	
	// Unsigned integer type suitable for radix sorting of keys
	typedef typename KeyConversion<K>::UnsignedBits ConvertedKeyType;

	
	/**
	 * Utility function: Returns the maximum problem size this enactor can sort on the device
	 * it was initialized for.
	 */
	static long long MaxProblemSize(const CudaProperties &props) 
	{
		long long element_size = (Base::KeysOnly()) ? sizeof(K) : sizeof(K) + sizeof(V);

		// Begin with device memory, subtract 128MB for video/spine/etc.  Factor in 
		// three vectors for keys (and values, if present)
		long long available_bytes = props.device_props.totalGlobalMem - 128;
		return available_bytes / (element_size * 3);
	}


protected:
	
	// Radix bits per pass
	static const int RADIX_BITS = 4;
	
	
	/**
	 * Utility function: Returns the default maximum number of threadblocks 
	 * this enactor class can launch.
	 */
	static int MaxGridSize(const CudaProperties &props, int max_grid_size = 0) 
	{
		if (max_grid_size == 0) {
			// No override: Fully populate all SMs
			max_grid_size = props.device_props.multiProcessorCount * 
					B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(props.kernel_ptx_version); 
		} 
		return max_grid_size;
	}
	
	
protected:
	
    /**
     * 
     */
    virtual cudaError_t PreSort(
    	SingleGridRadixSortStorage<K, V> &problem_storage,
    	int passes) 
    {
    	bool odd_passes = passes & 0x1;
    	
    	// Allocate device memory for temporary storage (if necessary)
    	if (problem_storage.d_keys[1] == NULL) {
    		cudaMalloc((void**) &problem_storage.d_keys[1], problem_storage.num_elements * sizeof(K));
    	}
    	if (odd_passes && (problem_storage.d_keys[2] == NULL)) {
    		cudaMalloc((void**) &problem_storage.d_keys[2], problem_storage.num_elements * sizeof(K));
    	}
    	if (!Base::KeysOnly() && (problem_storage.d_values[1] == NULL)) {
    		cudaMalloc((void**) &problem_storage.d_values[1], problem_storage.num_elements * sizeof(V));
    	}
    	if (odd_passes && !Base::KeysOnly() && (problem_storage.d_values[2] == NULL)) {
    		cudaMalloc((void**) &problem_storage.d_values[2], problem_storage.num_elements * sizeof(V));
    	}

    	return cudaSuccess;
    }
    
    
    /**
     * 
     */
    virtual cudaError_t PostSort(SingleGridRadixSortStorage<K, V> &problem_storage) 
    {
		return cudaSuccess;
    }


public:

	/**
	 * Constructor.
	 */
	SingleGridEnactor(
		int max_grid_size = 0,
		const CudaProperties &props = CudaProperties()) :
			Base::MultiCtaRadixSortingEnactor(
				MaxGridSize(props, max_grid_size),
				B40C_RADIXSORT_CYCLE_ELEMENTS(props.kernel_ptx_version , ConvertedKeyType, V),
				RADIX_BITS,
				props)
	{
		// Allocate synchronization counters
		cudaMalloc((void**) &d_sync, sizeof(int) * this->max_grid_size);
	}

	
	/**
     * Destructor
     */
    virtual ~SingleGridEnactor() 
    {
    	if (d_sync) cudaFree(d_sync);
    }
    
	
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <int LOWER_KEY_BITS>
	cudaError_t EnactSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
	{
		const int PASSES = (LOWER_KEY_BITS + RADIX_BITS - 1) / RADIX_BITS);

		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = this->max_grid_size;
		GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);
		
		// Compute number of spine elements to scan during this pass
		int spine_elements = grid_size * (1 << RADIX_BITS);
		int spine_cycles = (spine_elements + B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS - 1) / 
				B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
		spine_elements = spine_cycles * B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;

		// Perform any lazy allocation
		PreSort(problem_storage, PASSES);
		
		if (RADIXSORT_DEBUG)) {
    		printf("\ndevice_sm_version: %d, kernel_ptx_version: %d\n", 
    			this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version);
    		printf("Bottom-level reduction & scan kernels:\n\tgrid_size: %d, \n\tthreads: %d, \n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d\n\n",
    			grid_size, B40C_RADIXSORT_THREADS, this->tile_elements, work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
    		printf("Top-level spine scan:\n\tgrid_size: %d, \n\tthreads: %d, \n\tspine_block_elements: %d\n\n", 
    			grid_size, B40C_RADIXSORT_SPINE_THREADS, spine_elements);
    	}	
		
		// Uber kernel
		LsbRadixSortSmall<K, V, RADIX_BITS, PASSES, PreprocessKeyFunctor<K>, PostprocessKeyFunctor<K> >(
			d_sync,
			d_spine,
			(ConvertedKeyType **) problem_storage.d_keys,
			problem_storage.d_values[3],
			work_decomposition);
	    synchronize_if_enabled("ScanScatterDigits");

		// Perform any post-mortem
		PostSort(problem_storage);

		return cudaSuccess;
	}
	
	
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	cudaError_t EnactSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
	{
		return EnactSort<sizeof(K) * 8>(problem_storage);
	}
	
};







}// namespace b40c

