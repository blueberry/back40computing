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
 * Tuned Copy Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/arch_dispatch.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/copy/copy_enactor.cuh>
#include <b40c/copy/problem_config_tuned.cuh>
#include <b40c/copy/kernel_sweep.cuh>

namespace b40c {

/******************************************************************************
 * CopyEnactorTuned Declaration
 ******************************************************************************/

/**
 * Tuned copy enactor class.
 */
class CopyEnactorTuned : public copy::CopyEnactor
{
public:

	//---------------------------------------------------------------------
	// Helper Structures (need to be public for cudafe)
	//---------------------------------------------------------------------

	struct Storage;
	struct Detail;
	template <copy::ProbSizeGenre PROB_SIZE_GENRE> struct ConfigResolver;	

	
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Typedef for base class
	typedef copy::CopyEnactor BaseEnactorType;

	// Befriend our base types: they need to call back into a
	// protected methods (which are templated, and therefore can't be virtual)
	friend BaseEnactorType;


	//-----------------------------------------------------------------------------
	// Copy Operation
	//-----------------------------------------------------------------------------

    /**
	 * Performs a copy pass
	 */
	template <typename TunedConfig>
	cudaError_t CopyPass(
		typename TunedConfig::Sweep::T *d_dest,
		typename TunedConfig::Sweep::T *d_src,
		util::CtaWorkDistribution<typename TunedConfig::Sweep::SizeT> &work,
		int extra_bytes);


public:

	
	/**
	 * Constructor.
	 */
	CopyEnactorTuned() : BaseEnactorType::CopyEnactor() {}

	
	/**
	 * Enacts a copy on the specified device data using the specified
	 * granularity configuration
	 */
	using BaseEnactorType::Enact;

	
	/**
	 * Enacts a copy operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of bytes to be copied
	 * @param num_bytes
	 * 		Number of bytes to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <copy::ProbSizeGenre PROB_SIZE_GENRE>
	cudaError_t Enact(
		void *d_dest,
		void *d_src,
		size_t num_bytes,
		int max_grid_size = 0);


	/**
	 * Enacts a copy operation on the specified device data using
	 * a heuristic for selecting granularity configuration based upon
	 * problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of bytes to be copied
	 * @param num_bytes
	 * 		Number of bytes to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	cudaError_t Enact(
		void *d_dest,
		void *d_src,
		size_t num_bytes,
		int max_grid_size = 0);

};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
struct CopyEnactorTuned::Detail
{
	CopyEnactorTuned *enactor;
	int max_grid_size;

	// Constructor
	Detail(CopyEnactorTuned *enactor, int max_grid_size = 0) :
		enactor(enactor), max_grid_size(max_grid_size) {}
};


/**
 * Type for encapsulating storage details regarding an invocation
 */
struct CopyEnactorTuned::Storage
{
	void *d_dest;
	void *d_src;
	size_t num_bytes;

	// Constructor
	Storage(void *d_dest, void *d_src, size_t num_bytes) :
		d_dest(d_dest), d_src(d_src), num_bytes(num_bytes) {}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
 */
template <copy::ProbSizeGenre PROB_SIZE_GENRE>
struct CopyEnactorTuned::ConfigResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		// Obtain tuned granularity type
		typedef copy::TunedConfig<CUDA_ARCH, PROB_SIZE_GENRE> TunedConfig;
		typedef typename TunedConfig::Problem::T T;

		size_t num_elements = storage.num_bytes / sizeof(T);
		size_t extra_bytes = storage.num_bytes - (num_elements * sizeof(T));

		// Invoke base class enact with type
		return detail.enactor->template EnactInternal<TunedConfig, CopyEnactorTuned>(
			(T*) storage.d_dest, (T*) storage.d_src, num_elements, extra_bytes, detail.max_grid_size);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
 */
template <>
struct CopyEnactorTuned::ConfigResolver <copy::UNKNOWN>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename StorageType, typename DetailType>
	static cudaError_t Enact(StorageType &storage, DetailType &detail)
	{
		// Obtain large tuned granularity type
		typedef copy::TunedConfig<CUDA_ARCH, copy::LARGE> LargeConfig;
		typedef typename LargeConfig::Problem::T LargeT;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargeConfig::Sweep::TILE_ELEMENTS * 
			LargeConfig::Sweep::CTA_OCCUPANCY * 
			detail.enactor->SmCount();

		if (storage.num_bytes < saturating_load * sizeof(LargeT)) {

			// Invoke base class enact with small-problem config type

			typedef copy::TunedConfig<CUDA_ARCH, copy::SMALL> SmallConfig;
			typedef typename SmallConfig::Problem::T SmallT;

			size_t num_elements = storage.num_bytes / sizeof(SmallT);
			size_t extra_bytes = storage.num_bytes - (num_elements * sizeof(SmallT));

			return detail.enactor->template EnactInternal<SmallConfig, CopyEnactorTuned>(
				(SmallT*) storage.d_dest, (SmallT*) storage.d_src, num_elements, extra_bytes, detail.max_grid_size);
		}

		// Invoke base class enact with large-problem config type
		size_t num_elements = storage.num_bytes / sizeof(LargeT);
		size_t extra_bytes = storage.num_bytes - (num_elements * sizeof(LargeT));

		return detail.enactor->template EnactInternal<LargeConfig, CopyEnactorTuned>(
			(LargeT*) storage.d_dest, (LargeT*) storage.d_src, num_elements, extra_bytes, detail.max_grid_size);
	}
};


/******************************************************************************
 * CopyEnactorTuned Implementation
 ******************************************************************************/

/**
 * Performs a copy pass
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 */
template <typename TunedConfig>
cudaError_t CopyEnactorTuned::CopyPass(
	typename TunedConfig::Sweep::T *d_dest,
	typename TunedConfig::Sweep::T *d_src,
	util::CtaWorkDistribution<typename TunedConfig::Sweep::SizeT> &work,
	int extra_bytes)
{
	cudaError_t retval = cudaSuccess;

	int dynamic_smem = 0;

	// Sweep copy
	copy::TunedSweepCopyKernel<TunedConfig::PROB_SIZE_GENRE>
			<<<work.grid_size, TunedConfig::Sweep::THREADS, dynamic_smem>>>(
		(void *) d_src, (void *) d_dest, work, work_progress, extra_bytes);

	if (DEBUG) retval = util::B40CPerror(cudaThreadSynchronize(), "CopyEnactor SweepCopyKernel failed ", __FILE__, __LINE__);

	return retval;
}


/**
 * Enacts a copy operation on the specified device data using the
 * enumerated tuned granularity configuration
 */
template <copy::ProbSizeGenre PROB_SIZE_GENRE>
cudaError_t CopyEnactorTuned::Enact(
	void *d_dest,
	void *d_src,
	size_t num_bytes,
	int max_grid_size)
{
	Detail detail(this, max_grid_size);
	Storage storage(d_dest, d_src, num_bytes);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, ConfigResolver<PROB_SIZE_GENRE> >::Enact(
		storage, detail, PtxVersion());
}


/**
 * Enacts a copy operation on the specified device data using
 * a heuristic for selecting granularity configuration based upon
 * problem size.
 */
cudaError_t CopyEnactorTuned::Enact(
	void *d_dest,
	void *d_src,
	size_t num_bytes,
	int max_grid_size)
{
	return Enact<copy::UNKNOWN>(d_dest, d_src, num_bytes, max_grid_size);
}



} // namespace b40c

