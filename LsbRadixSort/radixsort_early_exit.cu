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
 * Early-exit Radix Sorting Enactor 
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <limits.h>

#include "b40c_error_synchronize.cu"
#include "radixsort_reduction_kernel.cu"
#include "radixsort_spine_kernel.cu"
#include "radixsort_scanscatter_kernel.cu"
#include "radixsort_multi_cta.cu"

namespace b40c {


/**
 * Early-exit radix sorting enactor class.  
 * 
 * It features "early-exit digit-passes": when the sorting operation 
 * detects that all keys have the same digit at the same digit-place, the pass 
 * for that digit-place is short-circuited, reducing the cost of that pass 
 * by 80%.  This makes our implementation suitable for even low-degree binning 
 * problems where sorting would normally be overkill.  
 * 
 * To use, simply create a specialized instance of this class with your 
 * key-type KeyType (and optionally value-type ValueType if sorting with satellite 
 * values).  E.g., for sorting signed ints:
 * 
 * 		EarlyExitLsbSortEnactor<int> sorting_enactor;
 * 
 * or for sorting floats paired with unsigned ints:
 * 			
 * 		EarlyExitLsbSortEnactor<float, unsigned int> sorting_enactor;
 * 
 * The enactor itself manages a small amount of device state for use when 
 * performing sorting operations.  To minimize GPU allocation overhead, 
 * enactors can be re-used over multiple sorting operations.  
 * 
 * The problem-storage for a sorting operation is independent of the sorting
 * enactor.  A single enactor can be reused to sort multiple instances of the
 * same type of problem storage.  The MultiCtaSortStorage structure
 * is used to manage the input/output/temporary buffers needed to sort 
 * a problem of a given size.  This enactor will lazily allocate any NULL
 * buffers contained within a problem-storage structure.  
 *
 * Sorting is invoked upon a problem-storage as follows:
 * 
 * 		sorting_enactor.EnactSort(device_storage);
 * 
 * This enactor will update the selector within the problem storage
 * to indicate which buffer contains the sorted output. E.g., 
 * 
 * 		device_storage.d_keys[device_storage.selector];
 * 
 * Please see the overview of MultiCtaSortStorage for more details.
 * 
 * 
 * @template-param KeyType
 * 		Type of keys to be sorted
 * @template-param ValueType
 * 		Type of values to be sorted.
 */
template <typename KeyType, typename ValueType = KeysOnly> 
class EarlyExitLsbSortEnactor : 
	
	public MultiCtaLsbSortEnactor<
		KeyType, 
		typename KeyConversion<KeyType>::UnsignedBits, 
		ValueType>
{
protected:

	//---------------------------------------------------------------------
	// Utility Types
	//---------------------------------------------------------------------

	// Unsigned integer type to cast keys as in order to make them suitable 
	// for radix sorting 
	typedef typename KeyConversion<KeyType>::UnsignedBits ConvertedKeyType;
	
	// Typedef for base class
	typedef MultiCtaLsbSortEnactor<KeyType, ConvertedKeyType, ValueType> Base;
	
    
	//---------------------------------------------------------------------
	// Default Granularity Parameterization
	//---------------------------------------------------------------------

	// Default granularity parameterization type.
	// 
	// We can tune this type per SM-architecture, per problem type.  Parameters
	// for separate kernels are largely performance-independent.
	// 
	template <int SM_ARCH, typename IndexType>
	struct DefaultSortingConfig
	{
		typedef typename MultiCtaLsbSortEnactor<KeyType, ConvertedKeyType, ValueType>::template MultiCtaConfig<

			//---------------------------------------------------------------------
			// Common
			//---------------------------------------------------------------------

			// IndexType
			IndexType,

			// RADIX_BITS
			(SM_ARCH >= 200) ? 					4 :		// 4-bit radix digits on GF100+
			(SM_ARCH >= 120) ? 					4 :		// 4-bit radix digits on GT200
												4,		// 4-bit radix digits on G80/90

			// LOG_SUBTILE_ELEMENTS
			(SM_ARCH >= 200) ? 					5 :		// 32 elements on GF100+
			(SM_ARCH >= 120) ? 					5 :		// 32 elements on GT200
												5,		// 32 elements on G80/90

			// CACHE_MODIFIER
			NONE,										// Default (CA: cache all levels)

			//---------------------------------------------------------------------
			// Upsweep Kernel
			//---------------------------------------------------------------------

			// UPSWEEP_CTA_OCCUPANCY
			(SM_ARCH >= 200) ? 					8 :		// 8 CTAs/SM on GF100+
			(SM_ARCH >= 120) ? 					5 :		// 5 CTAs/SM on GT200
												3,		// 3 CTAs/SM on G80/90
											
			// UPSWEEP_LOG_THREADS
			(SM_ARCH >= 200) ? 					7 :		// 128 threads/CTA on GF100+
			(SM_ARCH >= 120) ? 					7 :		// 128 threads/CTA on GT200
												7,		// 128 threads/CTA on G80/90

			// UPSWEEP_LOG_LOAD_VEC_SIZE
			(SM_ARCH >= 200) ? 					0 :		// vec-1 loads on GF100+
			(SM_ARCH >= 120) ? 					0 :		// vec-1 loads on GT200
												0,		// vec-1 loads on G80/90

			// UPSWEEP_LOG_LOADS_PER_TILE
			(SM_ARCH >= 200) ? 					2 :		// 4 loads/tile on GF100+
			(SM_ARCH >= 120) ? 					0 :		// 1 load/tile on GT200
												0,		// 1 load/tile on G80/90

			//---------------------------------------------------------------------
			// Spine-scan Kernel
			//---------------------------------------------------------------------

			// SPINE_CTA_OCCUPANCY
			1,											// 1 CTA/SM on all architectures

			// SPINE_LOG_THREADS
			(SM_ARCH >= 200) ? 					7 :		// 128 threads/CTA on GF100+
			(SM_ARCH >= 120) ? 					7 :		// 128 threads/CTA on GT200
												7,		// 128 threads/CTA on G80/90

			// SPINE_LOG_LOAD_VEC_SIZE
			(SM_ARCH >= 200) ? 					2 :		// vec-4 loads on GF100+
			(SM_ARCH >= 120) ? 					2 :		// vec-4 loads on GT200
												2,		// vec-4 loads on G80/90

			// SPINE_LOG_LOADS_PER_TILE
			(SM_ARCH >= 200) ? 					0 :		// 1 loads/tile on GF100+
			(SM_ARCH >= 120) ? 					0 :		// 1 loads/tile on GT200
												0,		// 1 loads/tile on G80/90

			// SPINE_LOG_RAKING_THREADS
			(SM_ARCH >= 200) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 0 :	// 1 warp on GF100+
			(SM_ARCH >= 120) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 0 :	// 1 warp on GT200
												B40C_LOG_WARP_THREADS(SM_ARCH) + 0,		// 1 warp on G80/90


			//---------------------------------------------------------------------
			// Downsweep Kernel
			//---------------------------------------------------------------------

			// DOWNSWEEP_CTA_OCCUPANCY
			(SM_ARCH >= 200) ? 					7 :		// 7 CTAs/SM on GF100+
			(SM_ARCH >= 120) ? 					5 :		// 5 CTAs/SM on GT200
												2,		// 2 CTAs/SM on G80/90

			// DOWNSWEEP_LOG_THREADS
			(SM_ARCH >= 200) ? 					7 :		// 128 threads/CTA on GF100+
			(SM_ARCH >= 120) ? 					7 :		// 128 threads/CTA on GT200
												7,		// 128 threads/CTA on G80/90

			// DOWNSWEEP_LOG_LOAD_VEC_SIZE
			(SM_ARCH >= 200) ? 					1 :		// vec-2 loads on GF100+
			(SM_ARCH >= 120) ? 					1 :		// vec-2 loads on GT200
												1,		// vec-2 loads on G80/90

			// DOWNSWEEP_LOG_LOADS_PER_CYCLE
			(SM_ARCH >= 200) ? 					1 :		// 2 loads/cycle on GF100+
			(SM_ARCH >= 120) ? 					0 :		// 1 load/cycle on GT200
												1,		// 2 loads/cycle on G80/90

			// DOWNSWEEP_LOG_CYCLES_PER_TILE
			(SM_ARCH >= 200) ?
				(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) > 4 ?
												0 : 	// 1 cycle/tile on GF100+ for large (64bit+) keys|values
												1) :	// 2 cycles/tile on GF100+
			(SM_ARCH >= 120) ?
				(B40C_MAX(sizeof(KeyType), sizeof(ValueType)) > 4 ?
												0 : 	// 1 cycle/tile on GT200 for large (64bit+) keys|values
												1) :	// 2 cycles/tile on GT200
												1,		// 2 cycles/tile on G80/90
											
			// DOWNSWEEP_LOG_RAKING_THREADS
			(SM_ARCH >= 200) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 1 :	// 2 warps on GF100+
			(SM_ARCH >= 120) ? 					B40C_LOG_WARP_THREADS(SM_ARCH) + 0 :	// 1 warp on GT200
												B40C_LOG_WARP_THREADS(SM_ARCH) + 2		// 4 warps on G80/90
		> Config;
	};

	
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------
		
	// Override for maximum grid size to launch for sweep kernels 
	// (0 == no override)
	int sweep_grid_size_override;

	// Pair of "selector" device integers.  The first selects the incoming device 
	// vector for even passes, the second selects the odd.
	int *d_selectors;
	

	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------
	
	/**
	 * Returns the number of threadblocks that the specified device should 
	 * launch for [up|down]sweep grids for the given problem size
	 */
	template <typename SortingConfig>
	int SweepGridSize(int num_elements) 
	{
		const int SUBTILE_ELEMENTS = 1 << SortingConfig::Upsweep::LOG_SUBTILE_ELEMENTS;

		int default_sweep_grid_size;
		if (this->cuda_props.device_sm_version < 120) {
			
			// G80/G90: Four times the SM-count
			default_sweep_grid_size = this->cuda_props.device_props.multiProcessorCount * 4;
			
		} else if (this->cuda_props.device_sm_version < 200) {
			
			// GT200: Special sauce
			
			// Start with with full downsweep occupancy of all SMs 
			default_sweep_grid_size = 
				this->cuda_props.device_props.multiProcessorCount * 
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

				if (default_sweep_grid_size <= this->cuda_props.device_props.multiProcessorCount) {
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

			// GF100: Hard-coded
			default_sweep_grid_size = 418;
		}
		
		// Reduce by override, if specified
		if (sweep_grid_size_override && (default_sweep_grid_size > sweep_grid_size_override)) {
			default_sweep_grid_size = sweep_grid_size_override;
		}

		// Reduce if we have less work than we can divide up among this 
		// many CTAs
		
		int subtiles = (num_elements + SUBTILE_ELEMENTS - 1) / SUBTILE_ELEMENTS;
		if (default_sweep_grid_size > subtiles) {
			default_sweep_grid_size = subtiles;
		}
		
		return default_sweep_grid_size;
	}	


	//-----------------------------------------------------------------------------
	// Sorting Operation 
	//-----------------------------------------------------------------------------
	
    /**
     * Post-sorting logic.
     */
	template <typename Storage> 
    cudaError_t PostSort(Storage &problem_storage, int passes) 
    {
/*	mooch		
    	// Save old selector
    	int old_selector = problem_storage.selector;
    	
    	// Copy out the selector from the last pass
		cudaMemcpy(
			&problem_storage.selector, 
			&d_selectors[passes & 0x1], 
			sizeof(int), 
			cudaMemcpyDeviceToHost);
		
		// Correct new selector if the original indicated that we started off from the alternate 
		problem_storage.selector ^= old_selector;
*/		
		
		return Base::PostSort(problem_storage, passes);
    }

	
    //-----------------------------------------------------------------------------
	// Sorting Pass 
	//-----------------------------------------------------------------------------

    /**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <
		typename SortingConfig,
		int CURRENT_PASS, 
		int CURRENT_BIT, 
		typename PreprocessFunctor, 
		typename PostprocessFunctor>
	
	cudaError_t DigitPlacePass(typename SortingConfig::Decomposition &work)
	{
		using namespace lsb_radix_sort::upsweep;
		using namespace lsb_radix_sort::scan;
		using namespace lsb_radix_sort::downsweep;

		typedef typename SortingConfig::Storage::IndexType IndexType; 

		int dynamic_smem, grid_size;

		// Detailed kernel granularity parameterization types
		typedef UpsweepKernelConfig <typename SortingConfig::Upsweep, PreprocessFunctor, CURRENT_PASS, CURRENT_BIT> 
			UpsweepKernelConfigType;
		typedef ScanKernelConfig <typename SortingConfig::SpineScan> 
			SpineScanKernelConfigType;
		typedef DownsweepKernelConfig <typename SortingConfig::Downsweep, PreprocessFunctor, PostprocessFunctor, CURRENT_PASS, CURRENT_BIT> 
			DownsweepKernelConfigType;

    	// Get kernel attributes
		cudaFuncAttributes upsweep_kernel_attrs, spine_scan_kernel_attrs, downsweep_attrs;
		cudaFuncGetAttributes(&upsweep_kernel_attrs, LsbRakingReductionKernel<UpsweepKernelConfigType>);
		cudaFuncGetAttributes(&spine_scan_kernel_attrs, LsbSpineScanKernel<SpineScanKernelConfigType>);
		cudaFuncGetAttributes(&downsweep_attrs, LsbScanScatterKernel<DownsweepKernelConfigType>);
		
		// mooch
		if (CURRENT_PASS == 0) {

			//
			// Invoke upsweep reduction kernel
			//
			
			grid_size = work.sweep_grid_size;
			
			dynamic_smem = 	(this->cuda_props.kernel_ptx_version >= 200) ? 	0 :  		// SM2.0+
							(this->cuda_props.kernel_ptx_version >= 120) ? 	downsweep_attrs.sharedSizeBytes - upsweep_kernel_attrs.sharedSizeBytes :			// GT200 gets the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels) 
																			0;			// SM1.0-1.1
			LsbRakingReductionKernel<UpsweepKernelConfigType>
				<<<grid_size, UpsweepKernelConfigType::THREADS, dynamic_smem>>>(
					d_selectors,
					(IndexType *) this->d_spine,
					(ConvertedKeyType *) work.problem_storage.d_keys[work.problem_storage.selector],
					(ConvertedKeyType *) work.problem_storage.d_keys[work.problem_storage.selector ^ 1],
					work);
			dbg_sync_perror_exit("EarlyExitLsbSortEnactor:: LsbRakingReductionKernel failed: ", __FILE__, __LINE__);

			//
			// Invoke spine scan kernel
			//
			
			grid_size = 	(this->cuda_props.kernel_ptx_version >= 200) ? 	1 :  						// SM2.0+ gets 1 CTA
							(this->cuda_props.kernel_ptx_version >= 120) ? 	work.sweep_grid_size :		// GT200 gets the same grid size as the sweep kernels 
																			work.sweep_grid_size;		// G80/90 gets the same grid size as the sweep kernels
	
			dynamic_smem = 	(this->cuda_props.kernel_ptx_version >= 200) ? 	0 :  		// SM2.0+
							(this->cuda_props.kernel_ptx_version >= 120) ? 	downsweep_attrs.sharedSizeBytes - spine_scan_kernel_attrs.sharedSizeBytes :			// GT200 gets the same smem allocation for every kernel launch (pad the reduction/top-level-scan kernels) 
																			0;			// SM1.0-1.1
			LsbSpineScanKernel<SpineScanKernelConfigType>
				<<<grid_size, SpineScanKernelConfigType::THREADS, dynamic_smem>>>(
					(IndexType *) this->d_spine,
					work.spine_elements);
			dbg_sync_perror_exit("EarlyExitLsbSortEnactor:: LsbSpineScanKernel failed: ", __FILE__, __LINE__);
			
			//
			// Invoke downsweep scan/scatter kernel
			//
			
			grid_size = work.sweep_grid_size;
			dynamic_smem = 0;
			
			LsbScanScatterKernel<DownsweepKernelConfigType>
				<<<grid_size, DownsweepKernelConfigType::THREADS, dynamic_smem>>>(
					d_selectors,
					(IndexType *) this->d_spine,
					(ConvertedKeyType *) work.problem_storage.d_keys[work.problem_storage.selector],
					(ConvertedKeyType *) work.problem_storage.d_keys[work.problem_storage.selector ^ 1],
					work.problem_storage.d_values[work.problem_storage.selector],
					work.problem_storage.d_values[work.problem_storage.selector ^ 1],
					work);
			dbg_sync_perror_exit("EarlyExitLsbSortEnactor:: LsbScanScatterKernel failed: ", __FILE__, __LINE__);
		}	
	    
		return cudaSuccess;
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
		int CURRENT_PASS, 
		int LAST_PASS, 
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses 
	{
		template <typename EnactorType>
		static void Invoke(EnactorType *enactor, typename SortingConfig::Decomposition &work) {
			// Invoke
			enactor->template DigitPlacePass<
				SortingConfig, 
				CURRENT_PASS, 
				CURRENT_BIT,
				NopFunctor<ConvertedKeyType>,
				NopFunctor<ConvertedKeyType> >(work);
			
			// Next
			UnrolledPasses<
				SortingConfig, 
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
		int LAST_PASS, 
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses <SortingConfig, 0, LAST_PASS, CURRENT_BIT, RADIX_BITS> 
	{
		template <typename EnactorType>
		static void Invoke(EnactorType *enactor, typename SortingConfig::Decomposition &work) {
			// Invoke
			enactor->template DigitPlacePass<
				SortingConfig, 
				0, 
				CURRENT_BIT,
				PreprocessKeyFunctor<KeyType>,
				NopFunctor<ConvertedKeyType> >(work);
			
			// Next
			UnrolledPasses<
				SortingConfig, 
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
		int LAST_PASS, 
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses <SortingConfig, LAST_PASS, LAST_PASS, CURRENT_BIT, RADIX_BITS> 
	{
		template <typename EnactorType>
		static void Invoke(EnactorType *enactor, typename SortingConfig::Decomposition &work) {
			// Invoke
			enactor->template DigitPlacePass<
				SortingConfig, 
				LAST_PASS, 
				CURRENT_BIT,
				NopFunctor<ConvertedKeyType>,
				PostprocessKeyFunctor<KeyType> >(work);
		}
	};

	/**
	 * Singular sorting pass (when there's only one pass).  Applies both 
	 * pre- and post-process bit-twiddling functors. 
	 */
	template <
		typename SortingConfig,
		int CURRENT_BIT, 
		int RADIX_BITS>
	struct UnrolledPasses <SortingConfig, 0, 0, CURRENT_BIT, RADIX_BITS> 
	{
		template <typename EnactorType>
		static void Invoke(EnactorType *enactor, typename SortingConfig::Decomposition &work) {
			// Invoke
			enactor->template DigitPlacePass<
				SortingConfig, 
				0, 
				CURRENT_BIT,
				PreprocessKeyFunctor<KeyType>,
				PostprocessKeyFunctor<KeyType> >(work);
		}
	};
	
	
	//-----------------------------------------------------------------------------
	// Architecture specialization 
	//-----------------------------------------------------------------------------

	// Sorting pass call-sites specialized per SM architecture 
	template <typename Storage, int SM_ARCH, int START_BIT, int NUM_BITS>
	struct Architecture
	{
		template<typename EnactorType>
		static cudaError_t EnactSort(EnactorType &enactor, Storage &problem_storage) 
		{
			typedef typename DefaultSortingConfig<SM_ARCH, Storage>::Config SortingConfig;
			return enactor.template EnactSort<SortingConfig, START_BIT, NUM_BITS>(problem_storage);
		}
	};
	
	// Host-side dispatch to specialized sorting pass call-sites 
	template <typename Storage, int START_BIT, int NUM_BITS>
	struct Architecture<Storage, 0, START_BIT, NUM_BITS> 
	{
		template<typename EnactorType>
		static cudaError_t EnactSort(EnactorType &enactor, Storage &problem_storage) 
		{
			// Determine the arch version of the we actually have a compiled kernel for
			int ptx_version = enactor.cuda_props.kernel_ptx_version;
			
			// Dispatch 
			if (ptx_version >= 200) {
				// SM2.0+
				return Architecture<Storage, 200, START_BIT, NUM_BITS>::EnactSort(enactor, problem_storage);
			
			} else if (ptx_version >= 120) {
				// SM1.2+
				return Architecture<Storage, 120, START_BIT, NUM_BITS>::EnactSort(enactor, problem_storage);
			
			} else {
				// SM1.0+
				return Architecture<Storage, 100, START_BIT, NUM_BITS>::EnactSort(enactor, problem_storage);
			}
		}
	};
	
	
public:

	//-----------------------------------------------------------------------------
	// Utility
	//-----------------------------------------------------------------------------

	/**
	 * Utility function: Returns the maximum problem size this enactor can sort on the device
	 * it was initialized for.
	 */
	size_t MaxProblemSize() 
	{
		// Begin with device memory, subtract 192MB for video/spine/etc.  Factor in 
		// two vectors for keys (and values, if present)

		size_t element_size = (Base::KeysOnly()) ? sizeof(KeyType) : sizeof(KeyType) + sizeof(ValueType);
		size_t available_bytes = this->cuda_props.device_props.totalGlobalMem - (192 * 1024 * 1024);
		size_t elements = available_bytes / (element_size * 2);
		
		return elements;
	}

	
	//-----------------------------------------------------------------------------
	// Construction 
	//-----------------------------------------------------------------------------

	/**
	 * Constructor.
	 */
	EarlyExitLsbSortEnactor(int sweep_grid_size_override = 0) :
		Base::MultiCtaLsbSortEnactor(),
		sweep_grid_size_override(sweep_grid_size_override),
		d_selectors(NULL)
	{
		// Allocate pair of ints denoting input and output vectors for even and odd passes
		cudaMalloc((void**) &d_selectors, 2 * sizeof(int));
	    dbg_perror_exit("EarlyExitLsbSortEnactor:: cudaMalloc d_selectors failed: ", __FILE__, __LINE__);
	}

	
	/**
     * Destructor
     */
    virtual ~EarlyExitLsbSortEnactor() 
    {
   		if (d_selectors) {
   			cudaFree(d_selectors);
   		    dbg_perror_exit("EarlyExitLsbSortEnactor:: cudaFree d_selectors failed: ", __FILE__, __LINE__);
   		}
    }
    
    
	//-----------------------------------------------------------------------------
	// Utility Sorting Interface 
    //
    // For generating sorting kernels having computational granularities in accordance 
    // with user-supplied granularity-specialization types.  (Useful for auto-tuning.) 
	//-----------------------------------------------------------------------------
    
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename SortingConfig, int START_BIT, int NUM_BITS> 
	cudaError_t EnactSort(typename SortingConfig::Storage &problem_storage) 
	{
		const int RADIX_BITS 			= SortingConfig::Upsweep::RADIX_BITS;
		const int NUM_PASSES 			= (NUM_BITS + RADIX_BITS - 1) / RADIX_BITS;
		const int SUBTILE_ELEMENTS 		= 1 << SortingConfig::Upsweep::LOG_SUBTILE_ELEMENTS;
		const int SPINE_TILE_ELEMENTS 	= 1 << 
				(SortingConfig::SpineScan::LOG_THREADS + 
				 SortingConfig::SpineScan::LOG_LOAD_VEC_SIZE + 
				 SortingConfig::SpineScan::LOG_LOADS_PER_TILE);
		
		const int OFFSET_BYTES = sizeof(typename SortingConfig::Storage::IndexType);

		int sweep_grid_size = SweepGridSize<SortingConfig>(problem_storage.num_elements);
		int spine_elements = Base::SpineElements(sweep_grid_size, RADIX_BITS, SPINE_TILE_ELEMENTS); 

		typename SortingConfig::Decomposition work(
			problem_storage,
			SUBTILE_ELEMENTS,
			sweep_grid_size,
			spine_elements);
			
		if (this->DEBUG) {
			printf("\n\n");
			printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n", 
				this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version);
			printf("Sorting: \t[radix_bits: %d, start_bit: %d, num_bits: %d, num_passes: %d, indexing-bits: %d]\n", 
				RADIX_BITS, START_BIT, NUM_BITS, NUM_PASSES, OFFSET_BYTES * 8);
			printf("Upsweep: \t[grid_size: %d, threads %d]\n",
				sweep_grid_size, 1 << SortingConfig::Upsweep::LOG_THREADS);
			printf("SpineScan: \t[grid_size: %d, threads %d, spine_elements: %d]\n",
				1, 1 << SortingConfig::SpineScan::LOG_THREADS, spine_elements);
			printf("Downsweep: \t[grid_size: %d, threads %d]\n",
				sweep_grid_size, 1 << SortingConfig::Downsweep::LOG_THREADS);
			printf("Work: \t\t[num_elements: %d, subtile_elements: %d, total_subtiles: %d, subtiles_per_cta: %d, extra_subtiles: %d]\n",
				work.num_elements, SUBTILE_ELEMENTS, work.total_subtiles, work.subtiles_per_cta, work.extra_subtiles);
			printf("\n\n");
		}
		
		// Perform any preparation prior to sorting
	    PreSort(problem_storage, spine_elements); 

	    // Perform sorting passes
	    UnrolledPasses<
			SortingConfig, 
			0, 
			NUM_PASSES - 1, 
			START_BIT, 
			RADIX_BITS>::Invoke(this, work);
		
		// Perform any cleanup after sorting
	    PostSort(problem_storage, NUM_PASSES);

	    return cudaSuccess;
	}

	
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename SortingConfig> 
	cudaError_t EnactSort(typename SortingConfig::Storage &problem_storage) 
	{
		return EnactSort<SortingConfig, 0, sizeof(KeyType) * 8>(problem_storage);
	}

	
	//-----------------------------------------------------------------------------
	// Primary Sorting Interface 
	//-----------------------------------------------------------------------------
	
	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Storage, int START_BIT, int NUM_BITS>
	cudaError_t EnactSort(Storage &problem_storage) 
	{
		return Architecture<Storage, __B40C_CUDA_ARCH__, START_BIT, NUM_BITS>::EnactSort(
			*this, problem_storage);
	}

	/**
	 * Enacts a radix sorting operation on the specified device data.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Storage>
	cudaError_t EnactSort(Storage &problem_storage) 
	{
		return EnactSort<Storage, 0, sizeof(KeyType) * 8>(problem_storage);
	}
	
};



}// namespace b40c

