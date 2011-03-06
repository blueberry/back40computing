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
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixsortTR.pdf):
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
 * Tuned Sort Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/work_distribution.cuh>
/*
#include <b40c/reduction/granularity_tuned.cuh>
#include <b40c/radix_sort/lsb_radix_sort_enactor.cuh>
#include <b40c/radix_sort/granularity.cuh>
#include <b40c/radix_sort/granularity_tuned.cuh>
*/

namespace b40c {

/******************************************************************************
 * LsbRadixsortEnactorTuned Declaration
 ******************************************************************************/

/**
 * Tuned sort enactor class.
 */
class LsbRadixsortEnactorTuned : public radix_sort::LsbRadixsortEnactor<LsbRadixsortEnactorTuned>
{
protected:
/*
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Typedefs for base classes
	typedef radix_sort::LsbRadixsortEnactor<LsbRadixsortEnactorTuned> BaseEnactorType;

	// Befriend our base types: they need to call back into a
	// protected methods (which are templated, and therefore can't be virtual)
	friend BaseEnactorType;
*/

	//-----------------------------------------------------------------------------
	// Sort Operation
	//-----------------------------------------------------------------------------

    /**
	 * Performs a sort pass
	 */
/*	
	template <typename TunedConfig>
	cudaError_t SortPass(
		typename TunedConfig::Upsweep::T *d_dest,
		typename TunedConfig::Upsweep::T *d_src,
		util::CtaWorkDistribution<typename TunedConfig::Upsweep::SizeT> &work,
		int spine_elements);
*/

	//-----------------------------------------------------------------------------
	// Granularity specialization interface required by ArchDispatch subclass
	//-----------------------------------------------------------------------------

	/**
	 * Dispatch call-back with static CUDA_ARCH
	 */
/*	
	template <int CUDA_ARCH, typename Storage, typename Detail>
	cudaError_t Enact(Storage &storage, Detail &detail);
*/

public:

	/**
	 * Constructor.
	 */	
	LsbRadixsortEnactorTuned();


	/**
	 * Enacts a sort operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be scanned
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to sort
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
/*	
	template <
		typename T,
		T BinaryOp(const T&, const T&),
		T Identity(),
		radix_sort::ProbSizeGenre PROB_SIZE_GENRE>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		size_t num_elements,
		int max_grid_size = 0);
*/

	/**
	 * Enacts a sort operation on the specified device data using
	 * a heuristic for selecting granularity configuration based upon
	 * problem size.
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be scanned
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to sort
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
/*	
	template <
		typename T,
		T BinaryOp(const T&, const T&),
		T Identity()>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		size_t num_bytes,
		int max_grid_size = 0);
*/		
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

namespace lsb_radix_sort_enactor_tuned {

/**
 * Type for encapsulating operational details regarding an invocation
 */
/*
template <typename LsbRadixsortEnactorTuned, typename SortProblemType>
struct Detail
{
	typedef SortProblemType ProblemType;
	
	LsbRadixsortEnactorTuned *enactor;
	int max_grid_size;

	// Constructor
	Detail() {}
	
	// Constructor
	Detail(LsbRadixsortEnactorTuned *enactor, int max_grid_size = 0) :
		enactor(enactor), max_grid_size(max_grid_size) {}
};
*/

/**
 * Type for encapsulating storage details regarding an invocation
 */
/*
template <typename T, typename SizeT>
struct Storage
{
	T *d_dest;
	T *d_src;
	SizeT num_elements;

	// Constructor
	Storage(T *d_dest, T *d_src, SizeT num_elements) :
		d_dest(d_dest), d_src(d_src), num_elements(num_elements) {}
};
*/

/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <radix_sort::ProbSizeGenre PROB_SIZE_GENRE>
struct ConfigResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
/*	
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		// Obtain tuned granularity type
		typedef radix_sort::TunedConfig<typename DetailType::ProblemType, CUDA_ARCH, PROB_SIZE_GENRE> TunedConfig;

		// Invoke base class enact with type
		return detail.enactor->template Enact<TunedConfig>(
			storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
	}
*/	
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct ConfigResolver <radix_sort::UNKNOWN>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
/*	
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		// Obtain large tuned granularity type
		typedef radix_sort::TunedConfig<typename DetailType::ProblemType, CUDA_ARCH, radix_sort::LARGE> LargeConfig;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargeConfig::Upsweep::TILE_ELEMENTS * LargeConfig::Upsweep::CTA_OCCUPANCY * detail.enactor->SmCount();
		if (storage.num_elements < saturating_load) {

			// Invoke base class enact with small-problem config type
			typedef radix_sort::TunedConfig<typename DetailType::ProblemType, CUDA_ARCH, radix_sort::SMALL> SmallConfig;
			return detail.enactor->template Enact<SmallConfig>(
				storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
		}

		// Invoke base class enact with type
		return detail.enactor->template Enact<LargeConfig>(
			storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
	}
*/	
};

} // namespace lsb_radix_sort_enactor_tuned


/******************************************************************************
 * LsbRadixsortEnactorTuned Implementation
 ******************************************************************************/

/**
 * Performs a sort pass
 */
/*
template <typename TunedConfig>
cudaError_t LsbRadixsortEnactorTuned::SortPass(
	typename TunedConfig::Upsweep::T *d_dest,
	typename TunedConfig::Upsweep::T *d_src,
	util::CtaWorkDistribution<typename TunedConfig::Upsweep::SizeT> &work,
	int spine_elements)
{
	typedef typename TunedConfig::Downsweep::ProblemType SortProblemType;
	typedef typename SortProblemType::T T;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			// No need to upsweep reduce or downsweep sort if there's only one CTA in the sweep grid
			int dynamic_smem = 0;
			TunedSpineScanKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<work.grid_size, TunedConfig::Spine::THREADS, dynamic_smem>>>(
				d_src, d_dest, work.num_elements);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "LsbRadixsortEnactor SpineScanKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option for dynamic smem allocation
			if (TunedConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, TunedUpsweepReductionKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"LsbRadixsortEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, TunedSpineScanKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"LsbRadixsortEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, TunedDownsweepScanKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"LsbRadixsortEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(
					upsweep_kernel_attrs.sharedSizeBytes,
					B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (TunedConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep reduce into spine
			TunedUpsweepReductionKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[0], TunedConfig::Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) d_spine, NULL, work, 0);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "LsbRadixsortEnactor TunedUpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

			// Spine scan
			TunedSpineScanKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[1], TunedConfig::Spine::THREADS, dynamic_smem[1]>>>(
				(T*) d_spine, (T*) d_spine, spine_elements);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "LsbRadixsortEnactor TunedSpineScanKernel failed ", __FILE__, __LINE__))) break;

			// Downsweep scan into spine
			TunedDownsweepScanKernel<SortProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[2], TunedConfig::Downsweep::THREADS, dynamic_smem[2]>>>(
				d_src, d_dest, (T*) d_spine, work);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "LsbRadixsortEnactor TunedDownsweepScanKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}
*/

/**
 * Constructor.
 */
/*
LsbRadixsortEnactorTuned::LsbRadixsortEnactorTuned()
	: BaseEnactorType::LsbRadixsortEnactor()
{
}
*/

/**
 * Enacts a sort operation on the specified device data using the
 * enumerated tuned granularity configuration
 */
/*
template <
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity(),
	radix_sort::ProbSizeGenre PROB_SIZE_GENRE>
cudaError_t LsbRadixsortEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	size_t num_elements,
	int max_grid_size)
{
	typedef size_t SizeT;
	typedef radix_sort::SortProblemType<T, SizeT, BinaryOp, Identity> ProblemType;
	typedef lsb_radix_sort_enactor_tuned::Detail<BaseEnactorType, ProblemType> Detail;			// Use base type pointer to ourselves
	typedef lsb_radix_sort_enactor_tuned::Storage<T, SizeT> Storage;
	typedef lsb_radix_sort_enactor_tuned::ConfigResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, max_grid_size);
	Storage storage(d_dest, d_src, num_elements);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(storage, detail, PtxVersion());
}
*/

/**
 * Enacts a sort operation on the specified device data using the
 * LARGE granularity configuration
 */
/*
template <
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity()>
cudaError_t LsbRadixsortEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	size_t num_elements,
	int max_grid_size)
{
	return Enact<T, BinaryOp, Identity, radix_sort::UNKNOWN>(d_dest, d_src, num_elements, max_grid_size);
}
*/


} // namespace b40c

