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
 * input.  (However, they may not necessarily point to the sorted output after 
 * sorting.)  The remaining pointers are for temporary storage arrays needed
 * to stream data between sorting passes.  These arrays can be allocated
 * lazily upon first sorting by the sorting enactor, or a-priori by the 
 * caller.  (If user-allocated, they should be large enough to accomodate 
 * num_elements.)    
 * 
 * It is the caller's responsibility to free any non-NULL storage arrays when
 * no longer needed.  This storage can be re-used for subsequent sorting 
 * operations of the same size.
 * 
 * NOTE: After a sorting operation has completed, the selecter member will
 * index the key (and value) pointers that contain the final sorted results.
 * (E.g., an odd number of sorting passes may leave the results in d_keys[1].)
 */
template <typename K, typename V = KeysOnlyType> 
struct EarlyExitRadixSortStorage
{
	// Pair of device vector pointers for keys
	K* d_keys[2];
	
	// Pair of device vector pointers for values
	V* d_values[2];

	// Number of elements for sorting in the above vectors 
	int num_elements;
	
	// Selector into the pair of device vector pointers indicating valid 
	// sorting elements (i.e., where the results are)
	int selector;

	// Constructor
	EarlyExitRadixSortStorage(int num_elements) :
		num_elements(num_elements), 
		selector(0) 
	{
		d_keys[0] = NULL;
		d_keys[1] = NULL;
		d_values[0] = NULL;
		d_values[1] = NULL;
	}

	// Constructor
	EarlyExitRadixSortStorage(int num_elements, K* keys, V* values = NULL) :
		num_elements(num_elements), 
		selector(0) 
	{
		d_keys[0] = keys;
		d_keys[1] = NULL;
		d_values[0] = values;
		d_values[1] = NULL;
	}
};





/**
 * Base class for early-exit, multi-CTA radix sorting enactors.
 */
template <typename K, typename V>
class BaseEarlyExitEnactor : public MultiCtaRadixSortingEnactor<K, V, EarlyExitRadixSortStorage<K, V> >
{
private:
	
	// Typedef for base class
	typedef MultiCtaRadixSortingEnactor<K, V, EarlyExitRadixSortStorage<K, V> > Base; 


protected:

	// Pair of "selector" device integers.  The first selects the incoming device 
	// vector for even passes, the second selects the odd.
	int *d_selectors;
	
	// Number of digit-place passes
	int passes;

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
		// two vectors for keys (and values, if present)
		long long available_bytes = props.device_props.totalGlobalMem - 128;
		return available_bytes / (element_size * 2);
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

			// No override: figure it out
			
			if (props.device_sm_version < 120) {
				
				// G80/G90
				max_grid_size = props.device_props.multiProcessorCount * 4;
				
			} else if (props.device_sm_version < 200) {
				
				// GT200 
				max_grid_size = props.device_props.multiProcessorCount * 
						B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(props.kernel_ptx_version); 

				// Increase by default every 64 million key-values
				int step = 1024 * 1024 * 64;		 
				max_grid_size *= (MaxProblemSize(props) + step - 1) / step;
				
			} else {

				// GF100
				max_grid_size = 418;
			}
		} 
		
		return max_grid_size;
	}
	
	
protected:

	
	/**
	 * Constructor.
	 */
	BaseEarlyExitEnactor(
		int passes,
		int max_radix_bits,
		int max_grid_size = 0,
		const CudaProperties &props = CudaProperties()) :
			Base::MultiCtaRadixSortingEnactor(
				MaxGridSize(props, max_grid_size),
				B40C_RADIXSORT_CYCLE_ELEMENTS(props.kernel_ptx_version , ConvertedKeyType, V),
				max_radix_bits,
				props), 
			d_selectors(NULL),
			passes(passes)
	{
		// Allocate pair of ints denoting input and output vectors for even and odd passes
		cudaMalloc((void**) &d_selectors, 2 * sizeof(int));
	}

	
	
	/**
	 * Determines the actual number of CTAs to launch for the given problem size
	 * 
	 * @return The actual number of CTAs that should be launched
	 */
	int GridSize(int num_elements)
	{
		// Initially assume that each threadblock will do only one 
		// tile worth of work (and that the last one will do any remainder), 
		// but then clamp it by the "max" restriction  

		int grid_size = num_elements / this->tile_elements;
		
		if (grid_size == 0) {

			// Always at least one block to process the remainder
			grid_size = 1;

		} else if (grid_size > this->max_grid_size) {
		
			int clamped_grid_size = this->max_grid_size;
		
			if (this->cuda_props.device_sm_version == 130) {
	
				clamped_grid_size = this->cuda_props.device_props.multiProcessorCount * 
						B40C_RADIXSORT_SCAN_SCATTER_CTA_OCCUPANCY(this->cuda_props.kernel_ptx_version); 

				// Increase by default every 64 million key-values
				int step = 1024 * 1024 * 64;		 
				clamped_grid_size *= (num_elements + step - 1) / step;

				// GT200 Fine-tune clamped_grid_size to avoid CTA camping 
				double multiplier1 = 4.0;
				double multiplier2 = 16.0;

				double delta1 = 0.068;
				double delta2 = 0.127;	

				int dividend = (num_elements + this->tile_elements - 1) / this->tile_elements;

				while(true) {

					double quotient = ((double) dividend) / (multiplier1 * clamped_grid_size);
					quotient -= (int) quotient;

					if ((quotient > delta1) && (quotient < 1 - delta1)) {

						quotient = ((double) dividend) / (multiplier2 * clamped_grid_size / 3.0);
						quotient -= (int) quotient;

						if ((quotient > delta2) && (quotient < 1 - delta2)) {
							break;
						}
					}

					if (clamped_grid_size == this->max_grid_size - 2) {
						// Bump it down by 30
						clamped_grid_size = this->max_grid_size - 30;
					} else {
						// Bump it down by 1
						clamped_grid_size -= 1;
					}
				}
			}
			
			grid_size = clamped_grid_size;
		}
		
		return grid_size;
	}
	

    /**
     * 
     */
    virtual cudaError_t PreSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
    {
    	// Allocate device memory for temporary storage (if necessary)
    	if (problem_storage.d_keys[1] == NULL) {
    		cudaMalloc((void**) &problem_storage.d_keys[1], problem_storage.num_elements * sizeof(K));
    	}
    	if (!Base::KeysOnly() && (problem_storage.d_values[1] == NULL)) {
    		cudaMalloc((void**) &problem_storage.d_values[1], problem_storage.num_elements * sizeof(V));
    	}

    	return cudaSuccess;
    }
    
    
    /**
     * 
     */
    virtual cudaError_t PostSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
    {
    	// Copy out the selector from the last pass
		cudaMemcpy(
			&problem_storage.selector, 
			&d_selectors[this->passes & 0x1], 
			sizeof(int), 
			cudaMemcpyDeviceToHost);

		return cudaSuccess;
    }

    
    /**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
	cudaError_t DigitPlacePass(
		const int grid_size,
		const EarlyExitRadixSortStorage<K, V> &problem_storage, 
		const CtaDecomposition &work_decomposition)
	{
		// Compute number of spine elements to scan during this pass
		int spine_elements = grid_size * (1 << RADIX_BITS);
		int spine_cycles = (spine_elements + B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS - 1) / 
				B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
		spine_elements = spine_cycles * B40C_RADIXSORT_SPINE_CYCLE_ELEMENTS;
		
		if (RADIXSORT_DEBUG && (PASS == 0)) {
    		
    		printf("\ndevice_sm_version: %d, kernel_ptx_version: %d\n", this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version);
    		printf("Bottom-level reduction & scan kernels:\n\tgrid_size: %d, \n\tthreads: %d, \n\tcycle_elements: %d, \n\tnum_big_blocks: %d, \n\tbig_block_elements: %d, \n\tnormal_block_elements: %d\n\textra_elements_last_block: %d\n\n",
    			grid_size, B40C_RADIXSORT_THREADS, this->tile_elements, work_decomposition.num_big_blocks, work_decomposition.big_block_elements, work_decomposition.normal_block_elements, work_decomposition.extra_elements_last_block);
    		printf("Top-level spine scan:\n\tgrid_size: %d, \n\tthreads: %d, \n\tspine_block_elements: %d\n\n", 
    			grid_size, B40C_RADIXSORT_SPINE_THREADS, spine_elements);
    	}	

    	// Get kernel attributes
		cudaFuncAttributes reduce_kernel_attrs, spine_scan_kernel_attrs, scan_scatter_attrs;
		cudaFuncGetAttributes(
			&reduce_kernel_attrs, 
			RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor>);
		cudaFuncGetAttributes(
			&spine_scan_kernel_attrs, 
			SrtsScanSpine<void>);
		cudaFuncGetAttributes(
			&scan_scatter_attrs, 
			ScanScatterDigits<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor>);


		//
		// Counting Reduction
		//

		// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
		if ((this->cuda_props.device_sm_version == 130) && 
				(work_decomposition.num_elements > this->cuda_props.device_props.multiProcessorCount * this->tile_elements * 2)) { 
			FlushKernel<void><<<grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
			synchronize_if_enabled("FlushKernel");
		}

		// GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
		int dynamic_smem = (this->cuda_props.kernel_ptx_version >= 130) ? 
			scan_scatter_attrs.sharedSizeBytes - reduce_kernel_attrs.sharedSizeBytes : 
			0;

		RakingReduction<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor> <<<grid_size, B40C_RADIXSORT_THREADS, dynamic_smem>>>(
			d_selectors,
			this->d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[0],
			(ConvertedKeyType *) problem_storage.d_keys[1],
			work_decomposition);
	    synchronize_if_enabled("RakingReduction");

		
		//
		// Spine
		//
		
		// GF100 and GT200 get the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels)
		dynamic_smem = (this->cuda_props.kernel_ptx_version >= 130) ? 
			scan_scatter_attrs.sharedSizeBytes - spine_scan_kernel_attrs.sharedSizeBytes : 
			0;
		
		SrtsScanSpine<void><<<grid_size, B40C_RADIXSORT_SPINE_THREADS, dynamic_smem>>>(
			this->d_spine,
			this->d_spine,
			spine_elements);
	    synchronize_if_enabled("SrtsScanSpine");

		
		//
		// Scanning Scatter
		//
		
		// Run tesla flush kernel if we have two or more threadblocks for each of the SMs
		if ((this->cuda_props.device_sm_version == 130) && 
				(work_decomposition.num_elements > this->cuda_props.device_props.multiProcessorCount * this->tile_elements * 2)) { 
			FlushKernel<void><<<grid_size, B40C_RADIXSORT_THREADS, scan_scatter_attrs.sharedSizeBytes>>>();
			synchronize_if_enabled("FlushKernel");
		}

		ScanScatterDigits<ConvertedKeyType, V, PASS, RADIX_BITS, BIT, PreprocessFunctor, PostprocessFunctor> <<<grid_size, B40C_RADIXSORT_THREADS, 0>>>(
			d_selectors,
			this->d_spine,
			(ConvertedKeyType *) problem_storage.d_keys[0],
			(ConvertedKeyType *) problem_storage.d_keys[1],
			problem_storage.d_values[0],
			problem_storage.d_values[1],
			work_decomposition);
	    synchronize_if_enabled("ScanScatterDigits");

		return cudaSuccess;
	}
	
	
	
public:

	
    /**
     * Destructor
     */
    virtual ~BaseEarlyExitEnactor() 
    {
   		if (d_selectors) cudaFree(d_selectors);
    }
    
};




/******************************************************************************
 * Sorting enactor classes
 ******************************************************************************/

/**
 * Generic sorting enactor class.  Simply create an instance of this class
 * with your key-type K (and optionally value-type V if sorting with satellite 
 * values).
 * 
 * Template specialization provides the appropriate enactor instance to handle 
 * the specified data types. 
 * 
 * @template-param K
 * 		Type of keys to be sorted
 *
 * @template-param V
 * 		Type of values to be sorted.
 *
 * @template-param ConvertedKeyType
 * 		Leave as default to effect necessary enactor specialization.
 */
template <typename K, typename V = KeysOnlyType, typename ConvertedKeyType = typename KeyConversion<K>::UnsignedBits>
class EarlyExitRadixSortingEnactor;



/**
 * Sorting enactor that is specialized for for 8-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned char> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(2, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = GridSize(problem_storage.num_elements);
		GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		PreSort(problem_storage);
		
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 

		PostSort(problem_storage);

		return cudaSuccess;
	}
};


/**
 * Sorting enactor that is specialized for for 16-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned short> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(4, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = GridSize(problem_storage.num_elements);
		GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		PreSort(problem_storage);
		
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<2, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<3, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 

		PostSort(problem_storage);

		return cudaSuccess;
	}
};


/**
 * Sorting enactor that is specialized for for 32-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned int> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(8, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = GridSize(problem_storage.num_elements);
		GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		PreSort(problem_storage);
		
		Base::template DigitPlacePass<0, 4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1, 4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<2, 4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<3, 4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<4, 4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<5, 4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<6, 4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<7, 4, 28, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (grid_size, problem_storage, work_decomposition); 

		PostSort(problem_storage);

		return cudaSuccess;
	}
};


/**
 * Sorting enactor that is specialized for for 64-bit key types
 */
template <typename K, typename V>
class EarlyExitRadixSortingEnactor<K, V, unsigned long long> : public BaseEarlyExitEnactor<K, V>
{
protected:

	typedef BaseEarlyExitEnactor<K, V> Base; 
	typedef typename Base::ConvertedKeyType ConvertedKeyType;

public:
	
	/**
	 * Constructor.
	 * 
	 * @param[in] 		max_grid_size  
	 * 		Maximum allowable number of CTAs to launch.  The default value of 0 indicates 
	 * 		that the dispatch logic should select an appropriate value for the target device.
	 */	
	EarlyExitRadixSortingEnactor(int max_grid_size = 0) : 
		Base::BaseEarlyExitEnactor(16, 4, max_grid_size) {}


	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSort(EarlyExitRadixSortStorage<K, V> &problem_storage) 
	{
		// Compute work distribution
		CtaDecomposition work_decomposition;
		int grid_size = GridSize(problem_storage.num_elements);
		GetWorkDecomposition(problem_storage.num_elements, grid_size, work_decomposition);

		PreSort(problem_storage);
		
		Base::template DigitPlacePass<0,  4, 0,  PreprocessKeyFunctor<K>,      NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<1,  4, 4,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<2,  4, 8,  NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<3,  4, 12, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<4,  4, 16, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<5,  4, 20, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<6,  4, 24, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<7,  4, 28, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<8,  4, 32, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition);
		Base::template DigitPlacePass<9,  4, 36, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<10, 4, 40, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<11, 4, 44, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<12, 4, 48, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<13, 4, 52, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<14, 4, 56, NopFunctor<ConvertedKeyType>, NopFunctor<ConvertedKeyType> >(grid_size, problem_storage, work_decomposition); 
		Base::template DigitPlacePass<15, 4, 60, NopFunctor<ConvertedKeyType>, PostprocessKeyFunctor<K> >    (grid_size, problem_storage, work_decomposition); 

		PostSort(problem_storage);

		return cudaSuccess;
	}
};


}// namespace b40c

