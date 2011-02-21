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
 * Tuned MemCopy Enactor
 ******************************************************************************/

#pragma once

#include <stdio.h>
#include <limits.h>

#include "memcopy_api_enactor.cuh"
#include "memcopy_api_granularity.cuh"
#include "memcopy_granularity_tuned_large.cuh"
#include "memcopy_granularity_tuned_small.cuh"

namespace b40c {

using namespace memcopy;


/******************************************************************************
 * Enumeration of predefined, tuned granularity configurations
 ******************************************************************************/

enum TunedGranularityEnum
{
	SMALL_PROBLEM,
	LARGE_PROBLEM			// default
};


// Forward declaration of tuned granularity configuration types
template <TunedGranularityEnum GRANULARITY_ENUM, int CUDA_ARCH> struct TunedGranularity;


/******************************************************************************
 * Forward declarations of memcopy kernel entry points that understand our
 * tuned granularity enumeration type.   TODO: Section can be removed if CUDA
 * Runtime is fixed to properly support template specialization around kernel
 * call sites.
 ******************************************************************************/

template <typename T, int GRANULARITY_ENUM>
__launch_bounds__ (
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__>::THREADS),
	(TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__>::OCCUPANCY))
__global__ void TunedMemcopyKernel(T * __restrict, T * __restrict, size_t * __restrict, CtaWorkDistribution<size_t>, int, int);




/******************************************************************************
 * Tuned Memcopy Enactor
 ******************************************************************************/

/**
 * Tuned memcopy enactor class.
 */
class MemcopyEnactorTuned :
	public MemcopyEnactor<MemcopyEnactorTuned>,
	public Architecture<__B40C_CUDA_ARCH__, MemcopyEnactorTuned>
{

protected:

	// Typedefs for base classes
	typedef MemcopyEnactor<MemcopyEnactorTuned> 					BaseEnactorType;
	typedef Architecture<__B40C_CUDA_ARCH__, MemcopyEnactorTuned> 	BaseArchType;

	// Our base classes are friends that invoke our templated
	// dispatch functions (which by their nature aren't virtual) 
	friend BaseEnactorType;
	friend BaseArchType;


	// Type for encapsulating operational details regarding an invocation
	template <TunedGranularityEnum _GRANULARITY_ENUM>
	struct Detail {
		static const TunedGranularityEnum GRANULARITY_ENUM 		= _GRANULARITY_ENUM;
		int max_grid_size;
		
		// Constructor
		Detail(int max_grid_size = 0) : max_grid_size(max_grid_size) {}
	};
	

	// Type for encapsulating storage details regarding an invocation
	struct Storage {
		void *d_dest;
		void *d_src;
		size_t num_bytes;

		// Constructor
		Storage(
			void *d_dest,
			void *d_src,
			size_t num_bytes) : d_dest(d_dest), d_src(d_src), num_bytes(num_bytes) {}
	};



	//-----------------------------------------------------------------------------
	// Memcopy Operation
	//
	// TODO: Section can be removed if CUDA Runtime is fixed to properly support
	// template specialization around kernel call sites.
	//-----------------------------------------------------------------------------

    /**
	 * Performs a memcopy pass
	 */
	template <typename MemcopyConfig>
	cudaError_t MemcopyPass(
		typename MemcopyConfig::T *d_dest,
		typename MemcopyConfig::T *d_src,
		CtaWorkDistribution<typename MemcopyConfig::SizeT> &work,
		int sweep_grid_size)
	{
		int dynamic_smem = 0;
		int threads = 1 << MemcopyConfig::LOG_THREADS;

		cudaError_t retval = cudaSuccess;
		do {

			TunedMemcopyKernel<typename MemcopyConfig::T, MemcopyConfig::GRANULARITY_ENUM>
					<<<sweep_grid_size, threads, dynamic_smem>>>(
				d_dest, d_src, d_work_progress, work, progress_selector, 0);

			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(),
				"MemcopyEnactorTuned:: MemcopyKernelTuned failed ", __FILE__, __LINE__))) break;

		} while (0);

		return retval;
	}


	//-----------------------------------------------------------------------------
	// Granularity specialization interface required by Architecture subclass
	//-----------------------------------------------------------------------------

	// Dispatch call-back with static CUDA_ARCH
	template <int CUDA_ARCH, typename Storage, typename Detail>
	cudaError_t Enact(Storage &storage, Detail &detail)
	{
		// Obtain tuned granularity type
		typedef TunedGranularity<Detail::GRANULARITY_ENUM, CUDA_ARCH> MemcopyConfig;
		typedef typename MemcopyConfig::T T;

		// Enact sort using that type
		return ((BaseEnactorType *) this)->template Enact<MemcopyConfig>(
			(T*) storage.d_dest, (T*) storage.d_src, storage.num_bytes, detail.max_grid_size);
	}

	
public:

	//-----------------------------------------------------------------------------
	// Construction 
	//-----------------------------------------------------------------------------

	/**
	 * Constructor.
	 */
	MemcopyEnactorTuned() : BaseEnactorType::MemcopyEnactor() {}

	
	//-----------------------------------------------------------------------------
	// Memcopy Interface
	//-----------------------------------------------------------------------------
	
	/**
	 * Enacts a memcopy operation on the specified device data using the
	 * enumerated tuned granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to array of bytes to be copied into
	 * @param d_src
	 * 		Pointer to array of bytes to be copied from
	 * @param num_bytes
	 * 		Number of bytes to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <TunedGranularityEnum GRANULARITY_ENUM>
	cudaError_t Enact(
		void *d_dest,
		void *d_src,
		size_t num_bytes,
		int max_grid_size = 0)
	{
		Detail<GRANULARITY_ENUM> detail(max_grid_size);
		Storage storage(d_dest, d_src, num_bytes);
		
		return BaseArchType::Enact(storage, detail);
	}

	/**
	 * Enacts a memcopy operation on the specified device data using the
	 * LARGE_PROBLEM granularity configuration
	 *
	 * @param d_dest
	 * 		Pointer to array of bytes to be copied into
	 * @param d_src
	 * 		Pointer to array of bytes to be copied from
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
		int max_grid_size = 0)
	{
		return Enact<LARGE_PROBLEM>(d_dest, d_src, num_bytes, max_grid_size);
	}
};




/******************************************************************************
 * Tuned granularity configuration types tagged by our tuning enumeration
 ******************************************************************************/

// Large-problem specialization of granularity config type
template <int CUDA_ARCH>
struct TunedGranularity<LARGE_PROBLEM, CUDA_ARCH>
	: large_problem_tuning::TunedConfig<CUDA_ARCH>
{
	static const TunedGranularityEnum GRANULARITY_ENUM 	= LARGE_PROBLEM;

	// Largely-unnecessary duplication of inner type data to accommodate
	// use in __launch_bounds__.   TODO: Section can be removed if CUDA Runtime is fixed to
	// properly support template specialization around kernel call sites.
	typedef large_problem_tuning::TunedConfig<CUDA_ARCH> Base;
	static const int THREADS 					= 1 << Base::LOG_THREADS;
	static const int OCCUPANCY 					= Base::CTA_OCCUPANCY;
};


// Small-problem specialization of granularity config type
template <int CUDA_ARCH>
struct TunedGranularity<SMALL_PROBLEM, CUDA_ARCH>
	: small_problem_tuning::TunedConfig<CUDA_ARCH>
{
	static const TunedGranularityEnum GRANULARITY_ENUM 	= SMALL_PROBLEM;

	// Largely-unnecessary duplication of inner type data to accommodate
	// use in __launch_bounds__.   TODO: Section can be removed if CUDA Runtime is fixed to
	// properly support template specialization around kernel call sites.
	typedef small_problem_tuning::TunedConfig<CUDA_ARCH> Base;
	static const int THREADS 					= 1 << Base::LOG_THREADS;
	static const int OCCUPANCY 					= Base::CTA_OCCUPANCY;
};




/******************************************************************************
 * Memcopy kernel entry points that understand our tuned granularity
 * enumeration type.  TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

template <typename T, int GRANULARITY_ENUM>
void TunedMemcopyKernel(
	T 				* __restrict d_out,
	T 				* __restrict d_in,
	size_t 			* __restrict d_work_progress,
	CtaWorkDistribution<size_t> work_decomposition,
	int 			progress_selector,
	int 			extra_bytes)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef TunedGranularity<(TunedGranularityEnum) GRANULARITY_ENUM, __B40C_CUDA_ARCH__> MemcopyConfig;
	typedef MemcopyKernelConfig<MemcopyConfig> KernelConfig;

	// Invoke the wrapped kernel logic
	MemcopyPass<KernelConfig, KernelConfig::WORK_STEALING>::Invoke(
		d_out, d_in, d_work_progress, work_decomposition, progress_selector, extra_bytes);
}




}// namespace b40c

