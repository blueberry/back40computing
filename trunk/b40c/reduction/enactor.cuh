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
 * Reduction enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/arch_dispatch.cuh>

#include <b40c/reduction/problem_type.cuh>
#include <b40c/reduction/autotuned_policy.cuh>
#include <b40c/reduction/upsweep/kernel.cuh>
#include <b40c/reduction/spine/kernel.cuh>

namespace b40c {
namespace reduction {


/**
 * Reduction enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for managing work-stealing progress
	// within a kernel invocation.
	util::CtaWorkProgressLifetime work_progress;

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;


	//-----------------------------------------------------------------------------
	// Helper structures
	//-----------------------------------------------------------------------------

	template <typename ProblemType, typename Enactor>
	friend class Detail;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
	 * Performs a reduction pass
	 */
	template <typename Policy, typename Detail>
	cudaError_t EnactPass(Detail &detail);


public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a reduction operation on the specified device data.  Uses
	 * a heuristic for selecting an autotuning policy based upon problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be reduced
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
		typename SizeT>
	cudaError_t Reduce(
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a reduction operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be reduced
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
		ProbSizeGenre PROB_SIZE_GENRE,
		typename SizeT>
	cudaError_t Reduce(
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a reduction on the specified device data.  Uses the specified
	 * kernel configuration policy.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be reduced
	 * @param num_elements
	 * 		Number of elements to reduce
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Policy>
	cudaError_t Reduce(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		typename Policy::SizeT num_elements,
		int max_grid_size = 0);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename ProblemType, typename Enactor>
struct Detail : ProblemType
{
	typedef typename ProblemType::T T;
	typedef typename ProblemType::SizeT SizeT;

	Enactor 	*enactor;
	T 			*d_dest;
	T 			*d_src;
	SizeT 		num_elements;
	int 		max_grid_size;

	// Constructor
	Detail(
		Enactor *enactor,
		T *d_dest,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0) :
			enactor(enactor),
			d_dest(d_dest),
			d_src(d_src),
			num_elements(num_elements),
			max_grid_size(max_grid_size)
	{}

	template <typename Policy>
	cudaError_t EnactPass()
	{
		return enactor->template EnactPass<Policy>(*this);
	}
};


/**
 * Helper structure for resolving and enacting autotuned policy
 */
template <ProbSizeGenre PROB_SIZE_GENRE>
struct PolicyResolver
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		// Obtain tuned granularity type
		typedef AutotunedPolicy<
			typename Detail::ProblemType,
			CUDA_ARCH,
			PROB_SIZE_GENRE> AutotunedPolicy;

		// Invoke enactor with type
		return detail.template EnactPass<AutotunedPolicy>();
	}
};


/**
 * Helper structure for resolving and enacting autotuned policy
 *
 * Specialization for UNKNOWN problem size genre to select other problem size
 * genres based upon problem size, machine width, etc.
 */
template <>
struct PolicyResolver <UNKNOWN_SIZE>
{
	/**
	 * ArchDispatch call-back with static CUDA_ARCH
	 */
	template <int CUDA_ARCH, typename Detail>
	static cudaError_t Enact(Detail &detail)
	{
		// Obtain large tuned granularity type
		typedef AutotunedPolicy<
			typename Detail::ProblemType,
			CUDA_ARCH,
			LARGE_SIZE> LargePolicy;

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::Upsweep::TILE_ELEMENTS *
			LargePolicy::Upsweep::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		if (detail.num_elements < saturating_load) {

			// Invoke enactor with small-problem config type
			typedef AutotunedPolicy<
				typename Detail::ProblemType,
				CUDA_ARCH,
				SMALL_SIZE> SmallPolicy;

			return detail.template EnactPass<SmallPolicy>();
		}

		// Invoke enactor with type
		return detail.template EnactPass<LargePolicy>();
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/

/**
 * Performs a reduction pass
 */
template <typename Policy, typename DetailType>
cudaError_t Enactor::EnactPass(DetailType &detail)
{
	typedef typename Policy::T 				T;
	typedef typename Policy::SizeT			SizeT;

	typedef typename Policy::Upsweep 		Upsweep;
	typedef typename Policy::Spine 			Spine;

	// Compute sweep grid size
	int sweep_grid_size = (Policy::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(detail.num_elements, detail.max_grid_size) :
		OccupiedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(detail.num_elements, detail.max_grid_size);

	// Use single-CTA kernel instead of multi-pass if problem is small enough
	if (detail.num_elements <= Spine::TILE_ELEMENTS * 3) {
		sweep_grid_size = 1;
	}

	// Compute spine elements: one element per CTA, rounded
	// up to nearest spine tile size
	int spine_elements = sweep_grid_size;

	// Obtain a CTA work distribution
	util::CtaWorkDistribution<SizeT> work;
	work.template Init<Upsweep::LOG_SCHEDULE_GRANULARITY>(detail.num_elements, sweep_grid_size);

	if (DEBUG) {
		if (sweep_grid_size > 1) {
			PrintPassInfo<Upsweep, Spine>(work, spine_elements);
		} else {
			PrintPassInfo<Spine>(work);
		}
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Make sure our spine is big enough
		if (retval = spine.Setup<T>(spine_elements)) break;

		// If we're work-stealing, make sure our work progress is set up
		// for the next pass
		if (Policy::Upsweep::WORK_STEALING) {
			if (retval = work_progress.Setup()) break;
		}

		if (work.grid_size == 1) {

			typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();

			SpineKernel<<<1, Spine::THREADS, 0>>>(
				detail.d_src, detail.d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

		} else {

			typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::UpsweepKernel();
			typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();

			int dynamic_smem[2] = 	{0, 0};
			int grid_size[2] = 		{work.grid_size, 1};

			// Tuning option: make sure all kernels have the same overall smem allocation
			if (Policy::UNIFORM_SMEM_ALLOCATION) if (retval = PadUniformSmem(dynamic_smem, UpsweepKernel, SpineKernel)) break;

			// Tuning option: make sure that all kernels launch the same number of CTAs)
			if (Policy::UNIFORM_GRID_SIZE) grid_size[1] = grid_size[0];

			// Upsweep reduction into spine
			UpsweepKernel<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				detail.d_src, (T*) spine(), work, work_progress);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Spine reduction
			SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(T*) spine(), detail.d_dest, spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	// Cleanup
	if (retval) {
		// We had an error, which means that the device counters may not be
		// properly initialized for the next pass: reset them.
		work_progress.HostReset();
	}

	return retval;
}


/**
 * Enacts a reduction on the specified device data.
 */
template <typename Policy>
cudaError_t Enactor::Reduce(
	typename Policy::T 			*d_dest,
	typename Policy::T 			*d_src,
	typename Policy::SizeT 		num_elements,
	int max_grid_size)
{
	Detail<Policy, Enactor> detail(
		this, d_dest, d_src, num_elements, max_grid_size);

	return EnactPass<Policy>(detail);
}


/**
 * Enacts a reduction operation on the specified device data.
 */
template <
	typename T,
	T BinaryOp(const T&, const T&),
	reduction::ProbSizeGenre PROB_SIZE_GENRE,
	typename SizeT>
cudaError_t Enactor::Reduce(
	T *d_dest,
	T *d_src,
	SizeT num_elements,
	int max_grid_size)
{
	typedef reduction::ProblemType<
		T,
		SizeT,
		BinaryOp> ProblemType;

	Detail<ProblemType, Enactor> detail(
		this, d_dest, d_src, num_elements, max_grid_size);

	return util::ArchDispatch<
		__B40C_CUDA_ARCH__,
		PolicyResolver<PROB_SIZE_GENRE> >::Enact(detail, PtxVersion());
}


/**
 * Enacts a reduction operation on the specified device data.
 */
template <
	typename T,
	T BinaryOp(const T&, const T&),
	typename SizeT>
cudaError_t Enactor::Reduce(
	T *d_dest,
	T *d_src,
	SizeT num_elements,
	int max_grid_size)
{
	return Reduce<T, BinaryOp, reduction::UNKNOWN_SIZE>(
		d_dest, d_src, num_elements, max_grid_size);
}


}// namespace reduction
}// namespace b40c

