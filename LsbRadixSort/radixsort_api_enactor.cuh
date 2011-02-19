/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * Base LSB Radix Sorting Enactor 
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <limits.h>

#include "b40c_kernel_utils.cuh"

#include "radixsort_kernel_upsweep.cuh"
#include "radixsort_kernel_spine.cuh"
#include "radixsort_kernel_downsweep.cuh"

namespace b40c {

using namespace lsb_radix_sort;


/**
 * Forward declaration of an internal utility type for managing the state for a
 * specific sorting operation.  TODO: Class can be moved inside LsbSortEnactor
 * if CUDA Runtime is fixed to properly support template specialization around
 * kernel call sites.
 */
template <typename Storage> struct SortingCtaDecomposition;



/**
 * Basic LSB radix sorting enactor class.  
 */
template <typename DerivedEnactorType = void>
class LsbSortEnactor 
{
protected:

	//---------------------------------------------------------------------
	// Specialize templated dispatch to self (no derived class) or 
	// derived class (CRTP -- curiously recurring template pattern)
	//---------------------------------------------------------------------

	template <typename DerivedType, int __dummy = 0>
	struct DispatchType 
	{
		typedef DerivedType Type;
	};
	
	template <int __dummy>
	struct DispatchType<void, __dummy>
	{
		typedef LsbSortEnactor<void> Type;
	};

	
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------
		
	// Number of bytes backed by d_spine 
	int spine_bytes;

	// Device properties
	const CudaProperties cuda_props;

	// Temporary device storage needed for scanning digit histograms produced
	// by separate CTAs
	void *d_spine;

	// Pair of "selector" device integers.  The first selects the incoming device 
	// vector for even passes, the second selects the odd.
	int *d_selectors;
	

	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------
	
	/**
	 * Utility function: Determines the number of spine elements needed 
	 * for a given grid size, rounded up to the nearest spine tile
	 */
	int SpineElements(int sweep_grid_size, int radix_bits, int spine_tile_elements) 
	{
		int spine_elements = sweep_grid_size * (1 << radix_bits);
		int spine_tiles = (spine_elements + spine_tile_elements - 1) / spine_tile_elements;
		return spine_tiles * spine_tile_elements;
	}

	
	/**
	 * Returns the number of threadblocks that the specified device should 
	 * launch for [up|down]sweep grids for the given problem size
	 */
	template <typename SortingConfig>
	int SweepGridSize(int num_elements, int max_grid_size) 
	{
		const int SCHEDULE_GRANULARITY = 1 << SortingConfig::Upsweep::LOG_SCHEDULE_GRANULARITY;

		int default_sweep_grid_size;
		if (cuda_props.device_sm_version < 120) {
			
			// G80/G90: Four times the SM-count
			default_sweep_grid_size = cuda_props.device_props.multiProcessorCount * 4;
			
		} else if (cuda_props.device_sm_version < 200) {
			
			// GT200: Special sauce
			
			// Start with with full downsweep occupancy of all SMs 
			default_sweep_grid_size = 
				cuda_props.device_props.multiProcessorCount * 
				SortingConfig::Downsweep::CTA_OCCUPANCY; 

			// Increase by default every 64 million key-values
			int step = 1024 * 1024 * 64;		 
			default_sweep_grid_size *= (num_elements + step - 1) / step;

			double multiplier1 = 4.0;
			double multiplier2 = 16.0;

			double delta1 = 0.068;
			double delta2 = 0.1285;
			
			int dividend = (num_elements + 512 - 1) / 512;

			int bumps = 0;
			while(true) {

				if (default_sweep_grid_size <= cuda_props.device_props.multiProcessorCount) {
					break;
				}
				
				double quotient = ((double) dividend) / (multiplier1 * default_sweep_grid_size);
				quotient -= (int) quotient;

				if ((quotient > delta1) && (quotient < 1 - delta1)) {

					quotient = ((double) dividend) / (multiplier2 * default_sweep_grid_size / 3.0);
					quotient -= (int) quotient;

					if ((quotient > delta2) && (quotient < 1 - delta2)) {
						break;
					}
				}

				if (bumps == 3) {
					// Bump it down by 27
					default_sweep_grid_size -= 27;
					bumps = 0;
				} else {
					// Bump it down by 1
					default_sweep_grid_size--;
					bumps++;
				}
			}
			
		} else {

			// GF10x
			if (cuda_props.device_sm_version == 210) {
				// GF110
				default_sweep_grid_size = 4 * (cuda_props.device_props.multiProcessorCount * SortingConfig::Downsweep::CTA_OCCUPANCY);
			} else {
				// Anything but GF110
				default_sweep_grid_size = 4 * (cuda_props.device_props.multiProcessorCount * SortingConfig::Downsweep::CTA_OCCUPANCY) - 2;
			}
		}
		
		// Reduce by override, if specified
		if (max_grid_size > 0) {
			default_sweep_grid_size = max_grid_size;
		}
		
		// Reduce if we have less work than we can divide up among this 
		// many CTAs
		
		int grains = (num_elements + SCHEDULE_GRANULARITY - 1) / SCHEDULE_GRANULARITY;
		if (default_sweep_grid_size > grains) {
			default_sweep_grid_size = grains;
		}
		
		return default_sweep_grid_size;
	}	


	//-----------------------------------------------------------------------------
	// Sorting Operation 
	//-----------------------------------------------------------------------------
	
    /**
     * Pre-sorting logic.
     */
	template <typename StorageType>
    cudaError_t PreSort(StorageType &problem_storage, int problem_spine_elements) 
    {
		typedef typename StorageType::KeyType KeyType;
		typedef typename StorageType::ValueType ValueType;

		cudaError_t retval = cudaSuccess;
		do {
			// If necessary, allocate pair of ints denoting input and output vectors for even and odd passes
			if (d_selectors == NULL) {
				if (retval = B40CPerror(cudaMalloc((void**) &d_selectors, 2 * sizeof(int)),
					"LsbSortEnactor cudaMalloc d_selectors failed", __FILE__, __LINE__)) break;
			}

			// If necessary, allocate device memory for temporary storage in the problem structure
			if (problem_storage.d_keys[0] == NULL) {
				if (retval = B40CPerror(cudaMalloc((void**) &problem_storage.d_keys[0], problem_storage.num_elements * sizeof(KeyType)),
					"LsbSortEnactor cudaMalloc problem_storage.d_keys[0] failed", __FILE__, __LINE__)) break;
			}
			if (problem_storage.d_keys[1] == NULL) {
				if (retval = B40CPerror(cudaMalloc((void**) &problem_storage.d_keys[1], problem_storage.num_elements * sizeof(KeyType)),
					"LsbSortEnactor cudaMalloc problem_storage.d_keys[1] failed", __FILE__, __LINE__)) break;
			}
			if (!IsKeysOnly<ValueType>()) {
				if (problem_storage.d_values[0] == NULL) {
					if (retval = B40CPerror(cudaMalloc((void**) &problem_storage.d_values[0], problem_storage.num_elements * sizeof(ValueType)),
						"LsbSortEnactor cudaMalloc problem_storage.d_values[0] failed", __FILE__, __LINE__)) break;
				}
				if (problem_storage.d_values[1] == NULL) {
					if (retval = B40CPerror(cudaMalloc((void**) &problem_storage.d_values[1], problem_storage.num_elements * sizeof(ValueType)),
						"LsbSortEnactor cudaMalloc problem_storage.d_values[1] failed", __FILE__, __LINE__)) break;
				}
			}

			// Make sure our spine is big enough
			int problem_spine_bytes = problem_spine_elements * sizeof(typename StorageType::SizeT);

			if (problem_spine_bytes > spine_bytes) {
				if (d_spine) {
					if (retval = B40CPerror(cudaFree(d_spine),
						"LsbSortEnactor cudaFree d_spine failed", __FILE__, __LINE__)) break;
				}

				spine_bytes = problem_spine_bytes;

				if (retval = B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
					"LsbSortEnactor cudaMalloc d_spine failed", __FILE__, __LINE__)) break;
			}
		} while (0);

    	return retval;
    }
	
	
	/**
     * Post-sorting logic.
     */
	template <typename Storage, typename SortingConfig>
    cudaError_t PostSort(Storage &problem_storage, int passes)
    {
		cudaError_t retval = cudaSuccess;

		if (!SortingConfig::Upsweep::EARLY_EXIT) {

			// We moved data between storage buffers at every pass
			problem_storage.selector = (problem_storage.selector + passes) & 0x1;

		} else {

			do {
				// Save old selector
				int old_selector = problem_storage.selector;

				// Copy out the selector from the last pass
				if (retval = B40CPerror(cudaMemcpy(&problem_storage.selector, &d_selectors[passes & 0x1], sizeof(int), cudaMemcpyDeviceToHost),
					"LsbSortEnactor cudaMemcpy d_selector failed", __FILE__, __LINE__)) break;

				// Correct new selector if the original indicated that we started off from the alternate
				problem_storage.selector ^= old_selector;

			} while (0);
		}

		return retval;
    }

	
    /**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <
		typename SortingConfig,
		typename Decomposition,
		int CURRENT_PASS, 
		int CURRENT_BIT, 
		typename PreprocessTraits, 
		typename PostprocessTraits>
	cudaError_t DigitPlacePass(Decomposition &work)
	{
		using namespace lsb_radix_sort::upsweep;
		using namespace lsb_radix_sort::spine_scan;
		using namespace lsb_radix_sort::downsweep;

		typedef typename Decomposition::SizeT SizeT;
		typedef typename SortingConfig::ConvertedKeyType ConvertedKeyType;

		// Detailed kernel granularity parameterization types
		typedef UpsweepKernelConfig <typename SortingConfig::Upsweep, PreprocessTraits, CURRENT_PASS, CURRENT_BIT> 
			UpsweepKernelConfigType;
		typedef SpineScanKernelConfig <typename SortingConfig::SpineScan> 
			SpineScanKernelConfigType;
		typedef DownsweepKernelConfig <typename SortingConfig::Downsweep, PreprocessTraits, PostprocessTraits, CURRENT_PASS, CURRENT_BIT> 
			DownsweepKernelConfigType;

		int dynamic_smem[3] = {0, 0, 0};
		int grid_size[3] = {work.sweep_grid_size, 1, work.sweep_grid_size};
		int threads[3] = {UpsweepKernelConfigType::THREADS, SpineScanKernelConfigType::THREADS, DownsweepKernelConfigType::THREADS};

		cudaError_t retval = cudaSuccess;

		do {
		
			// Tuning option for dynamic smem allocation
			if (SortingConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_scan_kernel_attrs, downsweep_attrs;

				if (retval = B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel<UpsweepKernelConfigType>),
					"LsbSortEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = B40CPerror(cudaFuncGetAttributes(&spine_scan_kernel_attrs, SpineScanKernel<SpineScanKernelConfigType>),
					"LsbSortEnactor cudaFuncGetAttributes spine_scan_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = B40CPerror(cudaFuncGetAttributes(&downsweep_attrs, DownsweepKernel<DownsweepKernelConfigType>),
					"LsbSortEnactor cudaFuncGetAttributes downsweep_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(
					upsweep_kernel_attrs.sharedSizeBytes,
					B40C_MAX(spine_scan_kernel_attrs.sharedSizeBytes, downsweep_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_scan_kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (SortingConfig::UNIFORM_GRID_SIZE) {

				// We need to make sure that all kernels launch the same number of CTAs
				grid_size[1] = work.sweep_grid_size;
			}

			// Invoke upsweep reduction kernel
			UpsweepKernel<UpsweepKernelConfigType>
				<<<grid_size[0], threads[0], dynamic_smem[0]>>>(
					d_selectors,
					(SizeT *) d_spine,
					(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector],
					(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector ^ 1],
					work);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(),
				"LsbSortEnactor:: UpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Invoke spine scan kernel
			SpineScanKernel<SpineScanKernelConfigType>
				<<<grid_size[1], threads[1], dynamic_smem[1]>>>(
					(SizeT *) d_spine,
					work.spine_elements);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(),
				"LsbSortEnactor:: SpineScanKernel failed ", __FILE__, __LINE__))) break;

			// Invoke downsweep scan/scatter kernel
			DownsweepKernel<DownsweepKernelConfigType>
				<<<grid_size[2], threads[2], dynamic_smem[2]>>>(
					d_selectors,
					(SizeT *) d_spine,
					(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector],
					(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector ^ 1],
					work.problem_storage->d_values[work.problem_storage->selector],
					work.problem_storage->d_values[work.problem_storage->selector ^ 1],
					work);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(),
				"LsbSortEnactor:: DownsweepKernel failed ", __FILE__, __LINE__))) break;

		} while (0);

		return retval;
	}

	
	//-----------------------------------------------------------------------------
	// Pass Unrolling 
	//-----------------------------------------------------------------------------

	/**
	 * Middle sorting passes (i.e., neither first, nor last pass).  Does not apply
	 * any pre/post bit-twiddling functors. 
	 */
	template <
		typename SortingConfig,
		typename Decomposition,
		int CURRENT_PASS, 
		int LAST_PASS, 
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses 
	{
		template <typename EnactorType>
		static cudaError_t Invoke(EnactorType *enactor, Decomposition &work)
		{
			// Invoke 
			cudaError_t retval = enactor->template DigitPlacePass<
				SortingConfig, 
				Decomposition,
				CURRENT_PASS, 
				CURRENT_BIT,
				KeyTraits<typename SortingConfig::ConvertedKeyType>,			// no bit twiddling
				KeyTraits<typename SortingConfig::ConvertedKeyType> >(work);	// no bit twiddling
			
			if (retval) return retval;

			// Next
			return UnrolledPasses<
				SortingConfig,
				Decomposition,
				CURRENT_PASS + 1, 
				LAST_PASS, 
				CURRENT_BIT + RADIX_BITS, 
				RADIX_BITS>::Invoke(enactor, work);
		}
	};
	
	/**
	 * First sorting pass (unless there's only one pass).  Applies the 
	 * appropriate pre-process bit-twiddling functor. 
	 */
	template <
		typename SortingConfig,
		typename Decomposition,
		int LAST_PASS, 
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses <SortingConfig, Decomposition, 0, LAST_PASS, CURRENT_BIT, RADIX_BITS> 
	{
		template <typename EnactorType>
		static cudaError_t Invoke(EnactorType *enactor, Decomposition &work)
		{
			// Invoke
			cudaError_t retval = enactor->template DigitPlacePass<
				SortingConfig, 
				Decomposition,
				0, 
				CURRENT_BIT,
				KeyTraits<typename Decomposition::KeyType>,						// possible bit twiddling
				KeyTraits<typename SortingConfig::ConvertedKeyType> >(work);	// no bit twiddling
			
			if (retval) return retval;

			// Next
			return UnrolledPasses<
				SortingConfig,
				Decomposition,
				1, 
				LAST_PASS, 
				CURRENT_BIT + RADIX_BITS, 
				RADIX_BITS>::Invoke(enactor, work);
		}
	};

	/**
	 * Last sorting pass (unless there's only one pass).  Applies the 
	 * appropriate post-process bit-twiddling functor. 
	 */
	template <
		typename SortingConfig,
		typename Decomposition,
		int LAST_PASS, 
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses <SortingConfig, Decomposition, LAST_PASS, LAST_PASS, CURRENT_BIT, RADIX_BITS> 
	{
		template <typename EnactorType>
		static cudaError_t Invoke(EnactorType *enactor, Decomposition &work)
		{
			// Invoke
			return enactor->template DigitPlacePass<
				SortingConfig, 
				Decomposition,
				LAST_PASS, 
				CURRENT_BIT,
				KeyTraits<typename SortingConfig::ConvertedKeyType>,		// no bit twiddling
				KeyTraits<typename Decomposition::KeyType> >(work);			// possible bit twiddling
		}
	};

	/**
	 * Singular sorting pass (when there's only one pass).  Applies both 
	 * pre- and post-process bit-twiddling functors. 
	 */
	template <
		typename SortingConfig,
		typename Decomposition,
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses <SortingConfig, Decomposition, 0, 0, CURRENT_BIT, RADIX_BITS> 
	{
		template <typename EnactorType>
		static cudaError_t Invoke(EnactorType *enactor, Decomposition &work)
		{
			// Invoke
			return enactor->template DigitPlacePass<
				SortingConfig, 
				Decomposition,
				0, 
				CURRENT_BIT,
				KeyTraits<typename Decomposition::KeyType>,				// possible bit twiddling
				KeyTraits<typename Decomposition::KeyType> >(work);		// possible bit twiddling
		}
	};
	
	
public:

	//---------------------------------------------------------------------
	// Utility Fields
	//---------------------------------------------------------------------
	
	// Debug level.  If set, the enactor blocks after kernel calls to check
	// for successful launch/execution
	bool DEBUG;

	//-----------------------------------------------------------------------------
	// Utility Methods
	//-----------------------------------------------------------------------------

	/**
	 * Utility function: Returns the maximum problem size this enactor can sort on the device
	 * it was initialized for.
	 */
	template <typename KeyType, typename ValueType>
	size_t MaxProblemSize() 
	{
		// Begin with device memory, subtract 192MB for video/spine/etc.  Factor in 
		// two vectors for keys (and values, if present)

		size_t element_size = (IsKeysOnly<ValueType>()) ? sizeof(KeyType) : sizeof(KeyType) + sizeof(ValueType);
		size_t available_bytes = cuda_props.device_props.totalGlobalMem - (192 * 1024 * 1024);
		size_t elements = available_bytes / (element_size * 2);
		
		return elements;
	}

	
	//-----------------------------------------------------------------------------
	// Construction 
	//-----------------------------------------------------------------------------

	/**
	 * Constructor.
	 */
	LsbSortEnactor() :
			d_selectors(NULL),
			d_spine(NULL),
			spine_bytes(0),
#if	defined(__THRUST_SYNCHRONOUS) || defined(DEBUG) || defined(_DEBUG)
			DEBUG(true) {}
#else
			DEBUG(false) {}
#endif
	
	/**
     * Destructor
     */
    virtual ~LsbSortEnactor() 
    {
   		if (d_spine) {
   			B40CPerror(cudaFree(d_spine), "LsbSortEnactor cudaFree d_spine failed: ", __FILE__, __LINE__);
   		}
   		if (d_selectors) {
   			B40CPerror(cudaFree(d_selectors), "LsbSortEnactor cudaFree d_selectors failed: ", __FILE__, __LINE__);
   		}
    }
    
    
	//-----------------------------------------------------------------------------
	// Basic Sorting Interface 
    //
    // For generating sorting kernels having computational granularities in accordance 
    // with user-supplied granularity-specialization types.  (Useful for auto-tuning.) 
	//-----------------------------------------------------------------------------
    
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @param problem_storage 
	 * 		Instance of MultiCtaSortStorage type describing the details of the 
	 * 		problem to sort. 
	 * @param max_grid_size
	 * 		Upper limit on the size of the grid of threadblocks that can be launched 
	 * 		per kernel.  (0 == use default)
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Storage, typename SortingConfig, int START_BIT, int NUM_BITS> 
	cudaError_t EnactSort(Storage &problem_storage, int max_grid_size = 0) 
	{
	    typedef SortingCtaDecomposition<Storage> Decomposition;

	    const int RADIX_BITS 			= SortingConfig::Upsweep::RADIX_BITS;
		const int NUM_PASSES 			= (NUM_BITS + RADIX_BITS - 1) / RADIX_BITS;
		const int SCHEDULE_GRANULARITY 	= 1 << SortingConfig::Upsweep::LOG_SCHEDULE_GRANULARITY;
		const int SPINE_TILE_ELEMENTS 	= 1 << 
				(SortingConfig::SpineScan::LOG_THREADS + 
				 SortingConfig::SpineScan::LOG_LOAD_VEC_SIZE + 
				 SortingConfig::SpineScan::LOG_LOADS_PER_TILE);
		
		int sweep_grid_size = SweepGridSize<SortingConfig>(problem_storage.num_elements, max_grid_size);
		int spine_elements = SpineElements(sweep_grid_size, RADIX_BITS, SPINE_TILE_ELEMENTS); 

		Decomposition work(&problem_storage, SCHEDULE_GRANULARITY, sweep_grid_size, spine_elements);
			
		if (DEBUG) {
			printf("\n\n");
			printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n", 
				cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
			printf("Sorting: \t[radix_bits: %d, start_bit: %d, num_bits: %d, num_passes: %d, indexing-bits: %d]\n", 
				RADIX_BITS, START_BIT, NUM_BITS, NUM_PASSES, sizeof(typename Storage::SizeT) * 8);
			printf("Upsweep: \t[grid_size: %d, threads %d]\n",
				sweep_grid_size, 1 << SortingConfig::Upsweep::LOG_THREADS);
			printf("SpineScan: \t[grid_size: %d, threads %d, spine_elements: %d]\n",
				1, 1 << SortingConfig::SpineScan::LOG_THREADS, spine_elements);
			printf("Downsweep: \t[grid_size: %d, threads %d]\n",
				sweep_grid_size, 1 << SortingConfig::Downsweep::LOG_THREADS);
			printf("Work: \t\t[num_elements: %d, schedule_granularity: %d, total_grains: %d, grains_per_cta: %d, extra_grains: %d]\n",
				work.num_elements, SCHEDULE_GRANULARITY, work.total_grains, work.grains_per_cta, work.extra_grains);
			printf("\n\n");
		}
		
		cudaError_t retval = cudaSuccess;
		do {

			// Perform any preparation prior to sorting
			if (retval = PreSort(problem_storage, spine_elements)) break;

			// Perform sorting passes
			if (retval = UnrolledPasses<
				SortingConfig,
				Decomposition,
				0,
				NUM_PASSES - 1,
				START_BIT,
				RADIX_BITS>::Invoke((typename DispatchType<DerivedEnactorType>::Type *) this, work)) break;

			// Perform any cleanup after sorting
			if (retval = PostSort<Storage, SortingConfig>(problem_storage, NUM_PASSES)) break;

		} while (0);

	    return retval;
	}
	
};



/**
 * Utility type for managing the state for a specific sorting operation
 */
template <typename Storage>
struct SortingCtaDecomposition : CtaWorkDistribution<typename Storage::SizeT>
{
	typedef typename Storage::KeyType KeyType;
	typedef typename Storage::ValueType ValueType;
	typedef typename Storage::SizeT SizeT;

	int sweep_grid_size;
	int spine_elements;
	Storage *problem_storage;

	// Constructor
	SortingCtaDecomposition(
		Storage *problem_storage,
		int schedule_granularity,
		int sweep_grid_size,
		int spine_elements) : CtaWorkDistribution<SizeT>(
				problem_storage->num_elements,
				schedule_granularity,
				sweep_grid_size),
			sweep_grid_size(sweep_grid_size),
			spine_elements(spine_elements),
			problem_storage(problem_storage) {};
};

}// namespace b40c

