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

#include <stdio.h>
#include <limits.h>

#include "reduction_api_enactor.cuh"
#include "reduction_api_granularity.cuh"
#include "reduction_granularity_tuned.cuh"

namespace b40c {

using namespace reduction;


/******************************************************************************
 * ReductionEnactorTuned Declaration
 ******************************************************************************/

/**
 * Tuned reduction enactor class.
 */
class ReductionEnactorTuned :
	public ReductionEnactor<ReductionEnactorTuned>,
	public Architecture<__B40C_CUDA_ARCH__, ReductionEnactorTuned>
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Typedefs for base classes
	typedef ReductionEnactor<ReductionEnactorTuned> 					BaseEnactorType;
	typedef Architecture<__B40C_CUDA_ARCH__, ReductionEnactorTuned> 	BaseArchType;

	// Befriend our base types: they need to call back into a
	// protected methods (which are templated, and therefore can't be virtual)
	friend BaseEnactorType;
	friend BaseArchType;


	//-----------------------------------------------------------------------------
	// Reduction Operation
	//-----------------------------------------------------------------------------

    /**
	 * Performs a reduction pass
	 */
	template <typename ReductionConfig>
	cudaError_t ReductionPass(
		typename ReductionConfig::Upsweep::Problem::T *d_dest,
		typename ReductionConfig::Upsweep::Problem::T *d_src,
		CtaWorkDistribution<typename ReductionConfig::Upsweep::Problem::SizeT> &work,
		int spine_elements);


	//-----------------------------------------------------------------------------
	// Granularity specialization interface required by Architecture subclass
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
		ProblemSize PROBLEM_SIZE>
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
 * ReductionEnactorTuned Implementation
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <ProblemSize _PROBLEM_SIZE, typename ReductionProblemType>
struct Detail
{
	typedef ReductionProblemType ProblemType;

	static const ProblemSize PROBLEM_SIZE = _PROBLEM_SIZE;

	int max_grid_size;

	// Constructor
	Detail(int max_grid_size = 0) : max_grid_size(max_grid_size) {}
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
 * Performs a reduction pass
 */
template <typename ReductionConfig>
cudaError_t ReductionEnactorTuned::ReductionPass(
	typename ReductionConfig::Upsweep::Problem::T *d_dest,
	typename ReductionConfig::Upsweep::Problem::T *d_src,
	CtaWorkDistribution<typename ReductionConfig::Upsweep::Problem::SizeT> &work,
	int spine_elements)
{
	typedef typename ReductionConfig::Upsweep::Problem Problem;
	typedef typename Problem::T T;

	cudaError_t retval = cudaSuccess;

	do {
		if (work.grid_size == 1) {

			// No need to scan the spine if there's only one CTA in the upsweep grid
			int dynamic_smem = 0;
			TunedUpsweepReductionKernel<Problem, ReductionConfig::PROBLEM_SIZE>
					<<<work.grid_size, ReductionConfig::Upsweep::THREADS, dynamic_smem>>>(
				d_src, d_dest, d_work_progress, work, progress_selector);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(), "ReductionEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[2] = 	{0, 0};
			int grid_size[2] = 		{work.grid_size, 1};

			// Tuning option for dynamic smem allocation
			if (ReductionConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs;
				if (retval = B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, TunedUpsweepReductionKernel<Problem, ReductionConfig::PROBLEM_SIZE>),
					"ReductionEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, TunedSpineReductionKernel<Problem, ReductionConfig::PROBLEM_SIZE>),
					"ReductionEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(upsweep_kernel_attrs.sharedSizeBytes, spine_kernel_attrs.sharedSizeBytes);

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (ReductionConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep reduction into spine
			TunedUpsweepReductionKernel<Problem, ReductionConfig::PROBLEM_SIZE>
					<<<grid_size[0], ReductionConfig::Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) d_spine, d_work_progress, work, progress_selector);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(), "ReductionEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

			// Spine reduction
			TunedSpineReductionKernel<Problem, ReductionConfig::PROBLEM_SIZE>
					<<<grid_size[1], ReductionConfig::Spine::THREADS, dynamic_smem[1]>>>(
				(T*) d_spine, d_dest, spine_elements);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(), "ReductionEnactor SpineReductionKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Dispatch call-back with static CUDA_ARCH
 */
template <int CUDA_ARCH, typename _Storage, typename _Detail>
cudaError_t ReductionEnactorTuned::Enact(_Storage &storage, _Detail &detail)
{
	// Obtain tuned granularity type
	typedef TunedConfig<typename _Detail::ProblemType, CUDA_ARCH, _Detail::PROBLEM_SIZE> ReductionConfig;

	// Invoke base class enact with type
	return BaseEnactorType::template Enact<ReductionConfig>(
		storage.d_dest, storage.d_src, storage.num_elements, detail.max_grid_size);
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
	ProblemSize PROBLEM_SIZE>
cudaError_t ReductionEnactorTuned::Enact(
	T *d_dest,
	T *d_src,
	size_t num_elements,
	int max_grid_size)
{
	typedef size_t SizeT;

	Detail<PROBLEM_SIZE, ReductionProblem<T, SizeT, BinaryOp, Identity> > detail(max_grid_size);
	Storage<T, SizeT> storage(d_dest, d_src, num_elements);

	return BaseArchType::Enact(storage, detail);
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
	// Hybrid approach
	if (sizeof(T) * num_elements > 256 * 1024) {
		// Large: problem size > 256KB
		return Enact<T, BinaryOp, Identity, LARGE>(d_dest, d_src, num_elements, max_grid_size);
	} else {
		// Small: problem size <= 256KB
		return Enact<T, BinaryOp, Identity, SMALL>(d_dest, d_src, num_elements, max_grid_size);
	}
}



} // namespace b40c

