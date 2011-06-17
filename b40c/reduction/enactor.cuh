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
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
	 * Performs a reduction pass
	 */
	template <typename Policy>
	cudaError_t EnactPass(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		util::CtaWorkDistribution<typename Policy::SizeT> &work,
		int spine_elements);

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
template <
	typename T,
	typename SizeT,
	T BinaryOp(const T&, const T&)>
struct Detail
{
	typedef reduction::ProblemType<T, SizeT, BinaryOp> ProblemType;

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
		typedef typename Detail::ProblemType ProblemType;

		// Obtain tuned granularity type
		typedef AutotunedPolicy<ProblemType, CUDA_ARCH, PROB_SIZE_GENRE> AutotunedPolicy;

		// Invoke enactor with type
		return detail.enactor->template Reduce<AutotunedPolicy>(
			detail.d_dest, detail.d_src, detail.num_elements, detail.max_grid_size);
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
		typedef typename Detail::ProblemType ProblemType;

		// Obtain large tuned granularity type
		typedef AutotunedPolicy<ProblemType, CUDA_ARCH, LARGE_SIZE> LargePolicy;

		// Identify the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::Upsweep::TILE_ELEMENTS *
			LargePolicy::Upsweep::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		if (detail.num_elements < saturating_load) {

			// Invoke enactor with small-problem config type
			typedef AutotunedPolicy<ProblemType, CUDA_ARCH, SMALL_SIZE> SmallPolicy;
			return detail.enactor->template Reduce<SmallPolicy>(
				detail.d_dest, detail.d_src, detail.num_elements, detail.max_grid_size);
		}

		// Invoke enactor with large-problem config type
		return detail.enactor->template Reduce<LargePolicy>(
			detail.d_dest, detail.d_src, detail.num_elements, detail.max_grid_size);
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/

/**
 * Performs a reduction pass
 */
template <typename Policy>
cudaError_t Enactor::EnactPass(
	typename Policy::T 									*d_dest,
	typename Policy::T 									*d_src,
	util::CtaWorkDistribution<typename Policy::SizeT> 	&work,
	int 												spine_elements)
{
	typedef typename Policy::Upsweep 		Upsweep;
	typedef typename Policy::Spine 			Spine;
	typedef typename Policy::T 				T;

	typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::UpsweepKernel();
	typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			SpineKernel<<<1, Spine::THREADS, 0>>>(
				d_src, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[2] = 	{0, 0};
			int grid_size[2] = 		{work.grid_size, 1};

			// Tuning option for dynamic smem allocation
			if (Policy::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel),
					"Enactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineKernel),
					"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(upsweep_kernel_attrs.sharedSizeBytes, spine_kernel_attrs.sharedSizeBytes);

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (Policy::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep reduction into spine
			UpsweepKernel<<<grid_size[0], Policy::Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) spine(), work, work_progress);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Spine reduction
			SpineKernel<<<grid_size[1], Policy::Spine::THREADS, dynamic_smem[1]>>>(
				(T*) spine(), d_dest, spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

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
	typedef typename Policy::Upsweep 	Upsweep;
	typedef typename Policy::Spine 		Spine;
	typedef typename Policy::T 			T;
	typedef typename Policy::SizeT		SizeT;

	// Compute sweep grid size
	int sweep_grid_size = (Policy::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size) :
		OccupiedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size);

	if (num_elements <= Spine::TILE_ELEMENTS * 2) {
		// No need to upsweep reduce if we can do it with a single spine
		// kernel in two or less sequential tiles (i.e., instead of two
		// back-to-back tiles where we would one tile per up/spine kernel)
		sweep_grid_size = 1;
	}

	// Compute spine elements: one element per CTA, rounded
	// up to nearest spine tile size
	int spine_elements = sweep_grid_size;

	// Obtain a CTA work distribution
	util::CtaWorkDistribution<SizeT> work;
	work.template Init<Upsweep::LOG_SCHEDULE_GRANULARITY>(num_elements, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version,
			cuda_props.kernel_ptx_version);
		if (sweep_grid_size > 1) {
			printf("Upsweep: \t[sweep_grid_size: %d, threads %d, SizeT %lu bytes, workstealing: %s, tile_elements: %d]\n",
				work.grid_size,
				Upsweep::THREADS,
				(unsigned long) sizeof(SizeT),
				Upsweep::WORK_STEALING ? "true" : "false", Upsweep::TILE_ELEMENTS);
			printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
				Spine::THREADS,
				spine_elements,
				Spine::TILE_ELEMENTS);
			printf("Work: \t\t[element bytes: %lu, num_elements: %lu, schedule_granularity: %d, total_grains: %lu, grains_per_cta: %lu extra_grains: %lu]\n",
				(unsigned long) sizeof(T),
				(unsigned long) work.num_elements,
				Upsweep::SCHEDULE_GRANULARITY,
				(unsigned long) work.total_grains,
				(unsigned long) work.grains_per_cta,
				(unsigned long) work.extra_grains);
		} else {
			printf("Spine: \t\t[threads: %d, tile_elements: %d]\n",
				Spine::THREADS, Spine::TILE_ELEMENTS);
		}
		fflush(stdout);
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

		// Invoke reduction pass
		if (retval = EnactPass<Policy>(d_dest, d_src, work, spine_elements)) break;

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
	typedef Detail<T, SizeT, BinaryOp> Detail;
	typedef PolicyResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, d_dest, d_src, num_elements, max_grid_size);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(
		detail, PtxVersion());
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

