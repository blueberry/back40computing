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
 * Consecutive-removal enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/arch_dispatch.cuh>

#include <b40c/consecutive_removal/problem_type.cuh>
#include <b40c/consecutive_removal/policy.cuh>
#include <b40c/consecutive_removal/autotuned_policy.cuh>
#include <b40c/consecutive_removal/downsweep/kernel.cuh>
#include <b40c/consecutive_removal/upsweep/kernel.cuh>
#include <b40c/scan/spine/kernel.cuh>

namespace b40c {
namespace consecutive_removal {


/**
 * Consecutive-removal enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
	 * Performs a consecutive removal pass
	 */
	template <typename Policy>
	cudaError_t EnactPass(
		typename Policy::T 			*d_dest,
		typename Policy::SizeT 		*d_num_compacted,
		typename Policy::T 			*d_src,
		util::CtaWorkDistribution<typename Policy::SizeT> &work,
		typename Policy::SizeT 		spine_elements);


public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a consecutive removal operation on the specified device data.  Uses
	 * a heuristic for selecting an autotuning policy based upon problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_num_compacted
	 * 		Pointer to result count
	 * @param d_src
	 * 		Pointer to array of elements to be trimmed
	 * @param num_elements
	 * 		Number of elements in d_src
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		typename SizeT>
	cudaError_t Trim(
		T *d_dest,
		SizeT *d_num_compacted,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a consecutive removal operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_num_compacted
	 * 		Pointer to result count
	 * @param d_src
	 * 		Pointer to array of elements to be trimmed
	 * @param num_elements
	 * 		Number of elements in d_src
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		ProbSizeGenre PROB_SIZE_GENRE,
		typename T,
		typename SizeT>
	cudaError_t Trim(
		T *d_dest,
		SizeT *d_num_compacted,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a consecutive removal on the specified device data.  Uses the specified
	 * kernel configuration policy.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_num_compacted
	 * 		Pointer to result count
	 * @param d_src
	 * 		Pointer to array of elements to be trimmed
	 * @param num_elements
	 * 		Number of elements in d_src
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Policy>
	cudaError_t Trim(
		typename Policy::T 		*d_dest,
		typename Policy::SizeT 	*d_num_compacted,
		typename Policy::T 		*d_src,
		typename Policy::SizeT 	num_elements,
		int 					max_grid_size = 0);
};



/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Type for encapsulating operational details regarding an invocation
 */
template <typename T, typename SizeT>
struct Detail
{
	typedef consecutive_removal::ProblemType<T, SizeT> ProblemType;

	Enactor 	*enactor;
	T 			*d_dest;
	SizeT		*d_num_compacted;
	T 			*d_src;
	SizeT 		num_elements;
	int 		max_grid_size;

	// Constructor
	Detail(
		Enactor *enactor,
		T *d_dest,
		SizeT *d_num_compacted,
		T *d_src,
		SizeT num_elements,
		int max_grid_size = 0) :
			enactor(enactor),
			d_dest(d_dest),
			d_num_compacted(d_num_compacted),
			d_src(d_src),
			num_elements(num_elements),
			max_grid_size(max_grid_size)
	{}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Default specialization for problem type genres
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
		return detail.enactor->template Trim<AutotunedPolicy>(
			detail.d_dest, detail.d_num_compacted, detail.d_src, detail.num_elements, detail.max_grid_size);
	}
};


/**
 * Helper structure for resolving and enacting tuning configurations
 *
 * Specialization for UNKNOWN problem type to select other problem type genres
 * based upon problem size, etc.
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

		// Identity the maximum problem size for which we can saturate loads
		int saturating_load = LargePolicy::Upsweep::TILE_ELEMENTS *
			LargePolicy::Upsweep::CTA_OCCUPANCY *
			detail.enactor->SmCount();

		if (detail.num_elements < saturating_load) {

			// Invoke enactor with small-problem config type
			typedef AutotunedPolicy<ProblemType, CUDA_ARCH, SMALL_SIZE> SmallPolicy;
			return detail.enactor->template Trim<SmallPolicy>(
				detail.d_dest, detail.d_num_compacted, detail.d_src, detail.num_elements, detail.max_grid_size);
		}

		// Invoke enactor with type
		return detail.enactor->template Trim<LargePolicy>(
			detail.d_dest, detail.d_num_compacted, detail.d_src, detail.num_elements, detail.max_grid_size);
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/


/**
 * Performs a consecutive removal pass
 */
template <typename Policy>
cudaError_t Enactor::EnactPass(
	typename Policy::T 				*d_dest,
	typename Policy::SizeT 			*d_num_compacted,
	typename Policy::T 				*d_src,
	util::CtaWorkDistribution<typename Policy::SizeT> &work,
	typename Policy::SizeT 			spine_elements)
{
	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			typedef typename Policy::Single 		Single;

			typename Policy::SingleKernelPtr SingleKernel = Policy::SingleKernel();

			SingleKernel<<<1, Single::THREADS, 0>>>(
				d_src, d_num_compacted, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SingleKernel failed ", __FILE__, __LINE__))) break;

		} else {

			typedef typename Policy::Upsweep 		Upsweep;
			typedef typename Policy::Spine 			Spine;
			typedef typename Policy::Downsweep 		Downsweep;

			typedef typename Spine::T 				SpineType;

			typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::UpsweepKernel();
			typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();
			typename Policy::DownsweepKernelPtr DownsweepKernel = Policy::DownsweepKernel();

			int dynamic_smem[3] = 	{0, 0, 0};
			int grid_size[3] = 		{work.grid_size, 1, work.grid_size};

			// Tuning option for dynamic smem allocation
			if (Policy::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs, downsweep_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepKernel),
					"Enactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineKernel),
					"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&downsweep_kernel_attrs, DownsweepKernel),
					"Enactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(
					upsweep_kernel_attrs.sharedSizeBytes,
					B40C_MAX(spine_kernel_attrs.sharedSizeBytes, downsweep_kernel_attrs.sharedSizeBytes));

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
				dynamic_smem[2] = max_static_smem - downsweep_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (Policy::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep scan into spine
			UpsweepKernel<<<grid_size[0], Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (SpineType*) spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Spine scan
			SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(SpineType*) spine(), (SpineType*) spine(), spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

			// Downsweep scan from spine
			DownsweepKernel<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				d_src, d_num_compacted, d_dest, (SpineType*) spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__))) break;

		}
	} while (0);

	return retval;
}


/**
 * Enacts a consecutive removal on the specified device data.
 */
template <typename Policy>
cudaError_t Enactor::Trim(
	typename Policy::T 				*d_dest,
	typename Policy::SizeT 			*d_num_compacted,
	typename Policy::T 				*d_src,
	typename Policy::SizeT			num_elements,
	int 							max_grid_size)
{
	typedef typename Policy::Upsweep 	Upsweep;
	typedef typename Policy::Spine 		Spine;
	typedef typename Policy::Downsweep 	Downsweep;
	typedef typename Policy::Single 	Single;

	typedef typename Policy::T 			T;
	typedef typename Policy::SizeT 		SizeT;
	typedef typename Spine::T 			SpineType;

	const int MIN_OCCUPANCY = B40C_MIN((int) Upsweep::CTA_OCCUPANCY, (int) Downsweep::CTA_OCCUPANCY);
	util::SuppressUnusedConstantWarning(MIN_OCCUPANCY);

	// Compute sweep grid size
	int sweep_grid_size = (Policy::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size) :
		OccupiedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size);

	if (num_elements <= Spine::TILE_ELEMENTS * 3) {
		// No need to upsweep reduce or downsweep if we can do it
		// with a single spine kernel in three or less sequential
		// tiles (i.e., instead of three back-to-back tiles where we would
		// do one tile per up/spine/down kernel)
		sweep_grid_size = 1;
	}

	// Compute spine elements (round up to nearest spine tile_elements)
	int spine_elements = ((sweep_grid_size + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

	// Obtain a CTA work distribution for copying items of type T
	util::CtaWorkDistribution<SizeT> work;
	work.template Init<Downsweep::LOG_SCHEDULE_GRANULARITY>(num_elements, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version,
			cuda_props.kernel_ptx_version);
		if (sweep_grid_size > 1) {
			printf("Upsweep: \t[sweep_grid_size: %d, threads %d, tile_elements: %d]\n",
				work.grid_size,
				Upsweep::THREADS,
				Upsweep::TILE_ELEMENTS);
			printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
				Spine::THREADS,
				spine_elements,
				Spine::TILE_ELEMENTS);
			printf("Downsweep: \t[sweep_grid_size: %d, threads %d, tile_elements: %d]\n",
				work.grid_size,
				Downsweep::THREADS,
				Downsweep::TILE_ELEMENTS);
			printf("Work: \t\t[element bytes: %lu, SizeT %lu bytes, num_elements: %lu, schedule_granularity: %d, total_grains: %lu, grains_per_cta: %lu, extra_grains: %lu]\n",
				(unsigned long) sizeof(T),
				(unsigned long) sizeof(SizeT),
				(unsigned long) work.num_elements,
				Downsweep::SCHEDULE_GRANULARITY,
				(unsigned long) work.total_grains,
				(unsigned long) work.grains_per_cta,
				(unsigned long) work.extra_grains);
		} else {
			printf("Single: \t[threads: %d, num_elements: %lu, tile_elements: %d]\n",
				Single::THREADS,
				(unsigned long) work.num_elements,
				Single::TILE_ELEMENTS);
		}
		fflush(stdout);
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Make sure our spine is big enough
		if (retval = spine.Setup<SpineType>(spine_elements)) break;

		// Invoke pass
		if (retval = EnactPass<Policy>(d_dest, d_num_compacted, d_src, work, spine_elements)) break;

	} while (0);

	return retval;
}


/**
 * Enacts a consecutive removal operation on the specified device.
 */
template <
	ProbSizeGenre PROB_SIZE_GENRE,
	typename T,
	typename SizeT>
cudaError_t Enactor::Trim(
	T 				*d_dest,
	SizeT 			*d_num_compacted,
	T 				*d_src,
	SizeT 			num_elements,
	int				max_grid_size)
{
	typedef Detail<T, SizeT> Detail;
	typedef PolicyResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, d_dest, d_num_compacted, d_src, num_elements, max_grid_size);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(
		detail, PtxVersion());
}


/**
 * Enacts a consecutive removal operation on the specified device data.
 */
template <
	typename T,
	typename SizeT>
cudaError_t Enactor::Trim(
	T 				*d_dest,
	SizeT 			*d_num_compacted,
	T 				*d_src,
	SizeT 			num_elements,
	int 			max_grid_size)
{
	return Trim<UNKNOWN_SIZE>(
		d_dest, d_src, num_elements, max_grid_size);
}



} // namespace consecutive_removal
} // namespace b40c

