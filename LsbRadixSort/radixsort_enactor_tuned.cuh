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
 * Tuned LSB Radix Sorting Enactor 
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <limits.h>

#include "radixsort_common.cuh"
#include "radixsort_enactor.cuh"
#include "radixsort_granularity.cuh"
#include "radixsort_granularity_tuned_large.cuh"
#include "radixsort_granularity_tuned_small.cuh"

namespace b40c {

using namespace lsb_radix_sort;


/******************************************************************************
 * Enumeration of pre-defined, tuned granularity configurations 
 ******************************************************************************/

enum TunedGranularityEnum
{
	SMALL_PROBLEM,
	LARGE_PROBLEM			// default
};


// Generic tuned granularity config type 
template <TunedGranularityEnum GRANULARITY_ENUM, int CUDA_ARCH, typename KeyType, typename ValueType, typename IndexType>
struct TunedGranularity;


// Large-problem specialization of granularity config type
template <int CUDA_ARCH, typename KeyType, typename ValueType, typename IndexType>
struct TunedGranularity<LARGE_PROBLEM, CUDA_ARCH, KeyType, ValueType, IndexType>
	: large_problem_tuning::TunedConfig<CUDA_ARCH, KeyType, ValueType, IndexType>
{
	static const TunedGranularityEnum GRANULARITY_ENUM 	= LARGE_PROBLEM;

	// Largely-unnecessary duplication of inner type data to accommodate
	// use in __launch_bounds__.  TODO: Section can be removed if CUDA Runtime
	// supports template specialization around kernel call sites.
	typedef large_problem_tuning::TunedConfig<CUDA_ARCH, KeyType, ValueType, IndexType> Base;
	static const int UPSWEEP_THREADS 					= 1 << Base::Upsweep::LOG_THREADS;
	static const int SPINESCAN_THREADS 					= 1 << Base::SpineScan::LOG_THREADS;
	static const int DOWNSWEEP_THREADS 					= 1 << Base::Downsweep::LOG_THREADS;
	static const int UPSWEEP_OCCUPANCY 					= Base::Upsweep::CTA_OCCUPANCY;
	static const int SPINESCAN_OCCUPANCY 				= Base::SpineScan::CTA_OCCUPANCY;
	static const int DOWNSWEEP_OCCUPANCY 				= Base::Downsweep::CTA_OCCUPANCY;
};


// Small-probelm specialization of granularity config type 
template <int CUDA_ARCH, typename KeyType, typename ValueType, typename IndexType>
struct TunedGranularity<SMALL_PROBLEM, CUDA_ARCH, KeyType, ValueType, IndexType>
	: small_problem_tuning::TunedConfig<CUDA_ARCH, KeyType, ValueType, IndexType>
{
	static const TunedGranularityEnum GRANULARITY_ENUM 	= SMALL_PROBLEM;

	// Largely-unnecessary duplication of inner type data to accommodate
	// use in __launch_bounds__.  TODO: Section can be removed if CUDA Runtime
	// supports template specialization around kernel call sites.
	typedef small_problem_tuning::TunedConfig<CUDA_ARCH, KeyType, ValueType, IndexType> Base;
	static const int UPSWEEP_THREADS 					= 1 << Base::Upsweep::LOG_THREADS;
	static const int SPINESCAN_THREADS 					= 1 << Base::SpineScan::LOG_THREADS;
	static const int DOWNSWEEP_THREADS 					= 1 << Base::Downsweep::LOG_THREADS;
	static const int UPSWEEP_OCCUPANCY 					= Base::Upsweep::CTA_OCCUPANCY;
	static const int SPINESCAN_OCCUPANCY 				= Base::SpineScan::CTA_OCCUPANCY;
	static const int DOWNSWEEP_OCCUPANCY 				= Base::Downsweep::CTA_OCCUPANCY;
};



/******************************************************************************
 * LSB sorting kernel entry points that understand our tuned granularity
 * enumeration type.  TODO: Section can be removed if CUDA Runtime supports template
 * specialization around kernel call sites.
 ******************************************************************************/

// Upsweep
template <typename KeyType, typename ConvertedKeyType, typename ValueType, typename IndexType, typename PreprocessTraits, typename PostprocessTraits, int CURRENT_PASS, int CURRENT_BIT, int GRANULARITY_ENUM>
__launch_bounds__ (
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType>::UPSWEEP_THREADS),
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType>::UPSWEEP_OCCUPANCY))
__global__ 
void TunedUpsweepKernel(
	int 						* d_selectors,
	IndexType 					* d_spine,
	ConvertedKeyType 			* d_in_keys,
	ConvertedKeyType			* d_out_keys,
	CtaDecomposition<IndexType>	work_decomposition)
{
	// Load the tuned granularity type identified by the enum for this architecture 
	using namespace upsweep; 
	typedef TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType> SortingConfig;
	typedef UpsweepKernelConfig<typename SortingConfig::Upsweep, PreprocessTraits, CURRENT_PASS, CURRENT_BIT> KernelConfig;
	
	// Invoke the wrapped kernel logic
	LsbUpsweep<KernelConfig>(d_selectors, d_spine, d_in_keys, d_out_keys, work_decomposition);
}

// SpineScan
template <typename KeyType, typename ValueType, typename IndexType, int GRANULARITY_ENUM>
__launch_bounds__ (
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType>::SPINESCAN_THREADS),
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType>::SPINESCAN_OCCUPANCY))
__global__ 
void TunedSpineScanKernel(
	int 		*d_spine,
	IndexType	spine_elements)
{
	// Load the tuned granularity type identified by the enum for this architecture 
	using namespace spine_scan; 
	typedef TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType> SortingConfig;
	typedef SpineScanKernelConfig<typename SortingConfig::SpineScan> KernelConfig;

	// Invoke the wrapped kernel logic
	LsbSpineScan<KernelConfig>(d_spine, spine_elements);
}

// Downsweep
template <typename KeyType, typename ConvertedKeyType, typename ValueType, typename IndexType, typename PreprocessTraits, typename PostprocessTraits, int CURRENT_PASS, int CURRENT_BIT, int GRANULARITY_ENUM>
__launch_bounds__ (
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType>::DOWNSWEEP_THREADS),
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType>::DOWNSWEEP_OCCUPANCY))
__global__ 
void TunedDownsweepKernel(
	int 						* __restrict d_selectors,
	IndexType 					* __restrict d_spine,
	ConvertedKeyType 			* __restrict d_keys0,
	ConvertedKeyType 			* __restrict d_keys1,
	ValueType 					* __restrict d_values0,
	ValueType					* __restrict d_values1,
	CtaDecomposition<IndexType>	work_decomposition)
{
	// Load the tuned granularity type identified by the enum for this architecture 
	using namespace downsweep; 
	typedef TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__, KeyType, ValueType, IndexType> SortingConfig;
	typedef DownsweepKernelConfig<typename SortingConfig::Downsweep, PreprocessTraits, PostprocessTraits, CURRENT_PASS, CURRENT_BIT> KernelConfig;
	
	// Invoke the wrapped kernel logic
	LsbDownsweep<KernelConfig>(d_selectors, d_spine, d_keys0, d_keys1, d_values0, d_values1, work_decomposition);
}



/******************************************************************************
 * Tuned LSB Sorting Enactor 
 ******************************************************************************/

/**
 * Tuned LSB radix sorting enactor class.  
 * 
 * @template-param KeyType
 * 		Type of keys to be sorted
 * @template-param ValueType
 * 		Type of values to be sorted.
 */
template <typename KeyType, typename ValueType = KeysOnly> 
class LsbSortEnactorTuned : 
	public LsbSortEnactor<KeyType, ValueType, LsbSortEnactorTuned<KeyType, ValueType> >,
	public Architecture<__B40C_CUDA_ARCH__, LsbSortEnactorTuned<KeyType, ValueType> >
{

protected:

	// Typedefs for base classes
	typedef LsbSortEnactor<KeyType, ValueType, LsbSortEnactorTuned<KeyType, ValueType> > BaseEnactorType;
	typedef Architecture<__B40C_CUDA_ARCH__, LsbSortEnactorTuned<KeyType, ValueType> > BaseArchType;

	// Our base classes are friends that invoke our templated
	// dispatch functions (which by their nature aren't virtual) 
	friend BaseEnactorType;
	friend BaseArchType;

	// Type for encapsulating static type data regarding a sorting invocation
	template <int _START_BIT, int _NUM_BITS, TunedGranularityEnum _GRANULARITY_ENUM>
	struct Detail {
		static const int START_BIT 								= _START_BIT;
		static const int NUM_BITS 								= _NUM_BITS;
		static const TunedGranularityEnum GRANULARITY_ENUM 		= _GRANULARITY_ENUM;
	};
	

	//-----------------------------------------------------------------------------
	// Sorting Operation 
	//
	// TODO: Section can be removed if CUDA Runtime supports template
	// specialization around kernel call sites.
	//-----------------------------------------------------------------------------
	
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
		typedef typename Decomposition::IndexType IndexType; 
		typedef typename SortingConfig::ConvertedKeyType ConvertedKeyType;

		int dynamic_smem[3] = {0, 0, 0};
		int grid_size[3] = {work.sweep_grid_size, 1, work.sweep_grid_size};
		int threads[3] = {1 << SortingConfig::Upsweep::LOG_THREADS, 1 << SortingConfig::SpineScan::LOG_THREADS, 1 << SortingConfig::Downsweep::LOG_THREADS};

		// Tuning option for dynamic smem allocation
		if (SortingConfig::UNIFORM_SMEM_ALLOCATION) {
			
			// We need to compute dynamic smem allocations to ensure all three 
			// kernels end up allocating the same amount of smem per CTA

	    	// Get kernel attributes
			cudaFuncAttributes upsweep_kernel_attrs, spine_scan_kernel_attrs, downsweep_attrs;
			cudaFuncGetAttributes(&upsweep_kernel_attrs, TunedUpsweepKernel<KeyType, ConvertedKeyType, ValueType, IndexType, PreprocessTraits, PostprocessTraits, CURRENT_PASS, CURRENT_BIT, SortingConfig::GRANULARITY_ENUM>);
			cudaFuncGetAttributes(&spine_scan_kernel_attrs, TunedSpineScanKernel<KeyType, ValueType, IndexType, SortingConfig::GRANULARITY_ENUM>);
			cudaFuncGetAttributes(&downsweep_attrs, TunedDownsweepKernel<KeyType, ConvertedKeyType, ValueType, IndexType, PreprocessTraits, PostprocessTraits, CURRENT_PASS, CURRENT_BIT, SortingConfig::GRANULARITY_ENUM>);

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
		TunedUpsweepKernel<KeyType, ConvertedKeyType, ValueType, IndexType, PreprocessTraits, PostprocessTraits, CURRENT_PASS, CURRENT_BIT, SortingConfig::GRANULARITY_ENUM>
			<<<grid_size[0], threads[0], dynamic_smem[0]>>>(
				this->d_selectors,
				(IndexType *) this->d_spine,
				(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector],
				(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector ^ 1],
				work);
		dbg_sync_perror_exit("LsbSortEnactor:: LsbRakingReductionKernel failed: ", __FILE__, __LINE__);

		// Invoke spine scan kernel
		TunedSpineScanKernel<KeyType, ValueType, IndexType, SortingConfig::GRANULARITY_ENUM>
			<<<grid_size[1], threads[1], dynamic_smem[1]>>>(
				(IndexType *) this->d_spine,
				work.spine_elements);
		dbg_sync_perror_exit("LsbSortEnactor:: LsbSpineScanKernel failed: ", __FILE__, __LINE__);

		// Invoke downsweep scan/scatter kernel
		TunedDownsweepKernel<KeyType, ConvertedKeyType, ValueType, IndexType, PreprocessTraits, PostprocessTraits, CURRENT_PASS, CURRENT_BIT, SortingConfig::GRANULARITY_ENUM>
			<<<grid_size[2], threads[2], dynamic_smem[2]>>>(
				this->d_selectors,
				(IndexType *) this->d_spine,
				(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector],
				(ConvertedKeyType *) work.problem_storage->d_keys[work.problem_storage->selector ^ 1],
				work.problem_storage->d_values[work.problem_storage->selector],
				work.problem_storage->d_values[work.problem_storage->selector ^ 1],
				work);
		dbg_sync_perror_exit("LsbSortEnactor:: LsbScanScatterKernel failed: ", __FILE__, __LINE__);

		return cudaSuccess;
	}

	
	//-----------------------------------------------------------------------------
	// Granularity specialization interface required by Architecture subclass
	//-----------------------------------------------------------------------------

	// The arch version of the code for the current device that actually have
	// compiled kernels for
	int PtxVersion()
	{
		return this->cuda_props.kernel_ptx_version;
	}

	// Dispatch call-back with static CUDA_ARCH
	template <int CUDA_ARCH, typename Storage, typename Detail>
	cudaError_t Enact(Storage &problem_storage)
	{
		typedef TunedGranularity<
			Detail::GRANULARITY_ENUM,
			CUDA_ARCH,
			KeyType,
			ValueType,
			typename Storage::IndexType> SortingConfig;

		return ((BaseEnactorType *) this)->template
			EnactSort<Storage, SortingConfig, Detail::START_BIT, Detail::NUM_BITS>(problem_storage);
	}

	
public:

	//-----------------------------------------------------------------------------
	// Construction 
	//-----------------------------------------------------------------------------

	/**
	 * Constructor.
	 */
	LsbSortEnactorTuned(int sweep_grid_size_override = 0) :
		BaseEnactorType::LsbSortEnactor(sweep_grid_size_override) {}

	
	//-----------------------------------------------------------------------------
	// Sorting Interface 
	//-----------------------------------------------------------------------------
	
	/**
	 * Enacts a radix sorting operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Storage, int START_BIT, int NUM_BITS, TunedGranularityEnum GRANULARITY_ENUM>
	cudaError_t EnactSort(Storage &problem_storage)
	{
		return BaseArchType::template
			Enact<Storage, Detail<START_BIT, NUM_BITS, GRANULARITY_ENUM> >(problem_storage);
	}

	/**
	 * Enacts a radix sorting operation on the specified device data using the
	 * LARGE_PROBLEM granularity configuration (i.e., tuned for large-problem sorting)
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Storage>
	cudaError_t EnactSort(Storage &problem_storage) 
	{
		return EnactSort<Storage, 0, sizeof(KeyType) * 8, LARGE_PROBLEM>(problem_storage);
	}
	
};



}// namespace b40c

