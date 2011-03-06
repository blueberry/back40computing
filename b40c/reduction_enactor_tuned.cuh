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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/******************************************************************************
 * Tuned Reduction Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/work_distribution.cuh>
#include <b40c/reduction/reduction_enactor.cuh>
#include <b40c/reduction/granularity.cuh>
#include <b40c/reduction/kernel_upsweep.cuh>
#include <b40c/reduction/kernel_spine.cuh>

namespace b40c {

/******************************************************************************
 * ReductionEnactorTuned Declaration
 ******************************************************************************/

/**
 * Tuned reduction enactor class.
 */
class ReductionEnactorTuned : public reduction::ReductionEnactor<ReductionEnactorTuned>
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Typedefs for base classes
	typedef reduction::ReductionEnactor<ReductionEnactorTuned> BaseEnactorType;

	// Befriend our base types: they need to call back into a
	// protected methods (which are templated, and therefore can't be virtual)
	friend BaseEnactorType;


	//-----------------------------------------------------------------------------
	// Reduction Operation
	//-----------------------------------------------------------------------------

    /**
	 * Performs a reduction pass
	 */
	template <typename TunedConfig>
	cudaError_t ReductionPass(
		typename TunedConfig::Upsweep::T *d_dest,
		typename TunedConfig::Upsweep::T *d_src,
		util::CtaWorkDistribution<typename TunedConfig::Upsweep::SizeT> &work,
		int spine_elements);


	//-----------------------------------------------------------------------------
	// Granularity specialization interface required by ArchDispatch subclass
	//-----------------------------------------------------------------------------

	/**
	 * Dispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Storage, typename Detail>
	cudaError_t Enact(Storage &storage, Detail &detail);


public:

	/**
	 * Constructor.
	 */
	ReductionEnactorTuned();


	/**
	 * Enacts a reduction operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be reduced
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to reduce
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		T BinaryOp(const T&, const T&),
		T Identity(),
		reduction::ProbSizeGenre PROB_SIZE_GENRE>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		size_t num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a reduction operation on the specified device data using
	 * a heuristic for selecting granularity configuration based upon
	 * problem size.
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be reduced
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to reduce
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		T BinaryOp(const T&, const T&),
		T Identity()>
	cudaError_t Enact(
		T *d_dest,
		T *d_src,
		size_t num_bytes,
		int max_grid_size = 0);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

namespace reduction_enactor_tuned {

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename ReductionEnactorTuned, typename ReductionProblemType>
struct Detail
{
	typedef ReductionProblemType ProblemType;
	
	ReductionEnactorTuned *enactor;
	int max_grid_size;

	// Constructor
	Detail(ReductionEnactorTuned *enactor, int max_grid_size = 0) :
		enactor(enactor), max_grid_size(max_grid_size) {}
};


/**
 * Type for encapsulating storage details regarding an invocation
 */
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


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <reduction::ProbSizeGenre PROB_SIZE_GENRE>
struct ConfigResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		// Obtain tuned granularity type
		typedef reduction::TunedConfig<typename DetailType::ProblemType, CUDA_ARCH, PROB_SIZE_GENRE> TunedConfig;

		// Invoke base class enact with type
		return detail.enactor->template Enact<TunedConfig>(
			storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct ConfigResolver <reduction::UNKNOWN>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		// Obtain large tuned granularity type
		typedef reduction::TunedConfig<typename DetailType::ProblemType, CUDA_ARCH, reduction::LARGE> LargeConfig;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargeConfig::Upsweep::TILE_ELEMENTS * LargeConfig::Upsweep::CTA_OCCUPANCY * detail.enactor->SmCount();
		if (storage.num_elements < saturating_load) {

			// Invoke base class enact with small-problem config type
			typedef reduction::TunedConfig<typename DetailType::ProblemType, CUDA_ARCH, reduction::SMALL> SmallConfig;
			return detail.enactor->template Enact<SmallConfig>(
				storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
		}

		// Invoke base class enact with type
		return detail.enactor->template Enact<LargeConfig>(
			storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
	}
};


} // namespace reduction_enactor_tuned


/******************************************************************************
 * ReductionEnactorTuned Implementation
 ******************************************************************************/

/**
 * Performs a reduction pass
 */
template <typename TunedConfig>
cudaError_t ReductionEnactorTuned::ReductionPass(
	typename TunedConfig::Upsweep::T *d_dest,
	typename TunedConfig::Upsweep::T *d_src,
	util::CtaWorkDistribution<typename TunedConfig::Upsweep::SizeT> &work,
	int spine_elements)
{
	using namespace reduction;

	typedef typename TunedConfig::Upsweep::ProblemType ReductionProblemType;
	typedef typename ReductionProblemType::T T;

	cudaError_t retval = cudaSuccess;

	do {
		if (work.grid_size == 1) {

			// No need to scan the spine if there's only one CTA in the upsweep grid
			int dynamic_smem = 0;
			TunedSpineReductionKernel<ReductionProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<work.grid_size, TunedConfig::Spine::THREADS, dynamic_smem>>>(
				d_src, d_dest, work.num_elements);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[2] = 	{0, 0};
			int grid_size[2] = 		{work.grid_size, 1};

			// Tuning option for dynamic smem allocation
			if (TunedConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, TunedUpsweepReductionKernel<ReductionProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ReductionEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, TunedSpineReductionKernel<ReductionProblemType, TunedConfig::PROB_SIZE_GENRE>),
					"ReductionEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(upsweep_kernel_attrs.sharedSizeBytes, spine_kernel_attrs.sharedSizeBytes);

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (TunedConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep reduction into spine
			TunedUpsweepReductionKernel<ReductionProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[0], TunedConfig::Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) d_spine, d_work_progress, work, progress_selector);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

			// Spine reduction
			TunedSpineReductionKernel<ReductionProblemType, TunedConfig::PROB_SIZE_GENRE>
					<<<grid_size[1], TunedConfig::Spine::THREADS, dynamic_smem[1]>>>(
				(T*) d_spine, d_dest, spine_elements);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor SpineReductionKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Constructor.
 */
ReductionEnactorTuned::ReductionEnactorTuned()
	: BaseEnactorType::ReductionEnactor()
{
}


/**
 * Enacts a reduction operation on the specified device data using the
 * enumerated tuned granularity configuration
 */
template <
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity(),
	reduction::ProbSizeGenre PROB_SIZE_GENRE>
cudaError_t ReductionEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	size_t num_elements,
	int max_grid_size)
{
	typedef size_t SizeT;
	typedef reduction::ReductionProblemType<T, SizeT, BinaryOp, Identity> ProblemType;
	typedef reduction_enactor_tuned::Detail<BaseEnactorType, ProblemType> Detail;			// Use base type pointer to ourselves
	typedef reduction_enactor_tuned::Storage<T, SizeT> Storage;
	typedef reduction_enactor_tuned::ConfigResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, max_grid_size);
	Storage storage(d_dest, d_src, num_elements);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(storage, detail, PtxVersion());
}


/**
 * Enacts a reduction operation on the specified device data using the
 * LARGE granularity configuration
 */
template <
	typename T,
	T BinaryOp(const T&, const T&),
	T Identity()>
cudaError_t ReductionEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	size_t num_elements,
	int max_grid_size)
{
	return Enact<T, BinaryOp, Identity, reduction::UNKNOWN>(d_dest, d_src, num_elements, max_grid_size);
}



} // namespace b40c

