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
 * Segmented scan enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/arch_dispatch.cuh>

#include <b40c/segmented_scan/problem_type.cuh>
#include <b40c/segmented_scan/policy.cuh>
#include <b40c/segmented_scan/autotuned_policy.cuh>
#include <b40c/segmented_scan/downsweep/kernel.cuh>
#include <b40c/segmented_scan/spine/kernel.cuh>
#include <b40c/segmented_scan/upsweep/kernel.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Segmented scan enactor class.
 */
class Enactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine partial_spine;

	// Temporary device storage needed for reducing partials produced by separate CTAs
	util::Spine flag_spine;

	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

    /**
	 * Performs a segmented scan pass
	 */
	template <typename Policy>
	cudaError_t EnactPass(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		typename Policy::Flag *d_flag_src,
		util::CtaWorkDistribution<typename Policy::SizeT> &work,
		typename Policy::SizeT spine_elements);


public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a segmented scan operation on the specified device data.  Uses
	 * a heuristic for selecting an autotuning policy based upon problem size.
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param d_flag_src
	 * 		Pointer to array of "head flags" that demarcate independent scan segments
	 * @param num_elements
	 * 		Number of elements to segmented scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&),
		T Identity(),
		typename Flag,
		typename SizeT>
	cudaError_t Scan(
		T *d_dest,
		T *d_src,
		Flag *d_flag_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a segmented scan operation on the specified device data.  Uses the
	 * specified problem size genre enumeration to select autotuning policy.
	 *
	 * (Using this entrypoint can save compile time by not compiling tuned
	 * kernels for each problem size genre.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param d_flag_src
	 * 		Pointer to array of "head flags" that demarcate independent scan segments
	 * @param num_elements
	 * 		Number of elements to segmented scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <
		typename T,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&),
		T Identity(),
		ProbSizeGenre PROB_SIZE_GENRE,
		typename Flag,
		typename SizeT>
	cudaError_t Scan(
		T *d_dest,
		T *d_src,
		Flag *d_flag_src,
		SizeT num_elements,
		int max_grid_size = 0);


	/**
	 * Enacts a segmented scan on the specified device data.  Uses the specified
	 * kernel configuration policy.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to result location
	 * @param d_src
	 * 		Pointer to array of elements to be scanned
	 * @param d_flag_src
	 * 		Pointer to array of "head flags" that demarcate independent scan segments
	 * @param num_elements
	 * 		Number of elements to segmented scan
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename Policy>
	cudaError_t Scan(
		typename Policy::T *d_dest,
		typename Policy::T *d_src,
		typename Policy::Flag *d_flag_src,
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
	typename Flag,
	typename SizeT,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity()>
struct Detail
{
	typedef segmented_scan::ProblemType<T, Flag, SizeT, EXCLUSIVE, BinaryOp, Identity> ProblemType;

	Enactor 	*enactor;
	T 			*d_dest;
	T 			*d_src;
	Flag		*d_flag_src;
	SizeT 		num_elements;
	int 		max_grid_size;

	// Constructor
	Detail(
		Enactor *enactor,
		T *d_dest,
		T *d_src,
		Flag *d_flag_src,
		SizeT num_elements,
		int max_grid_size = 0) :
			enactor(enactor),
			d_dest(d_dest),
			d_src(d_src),
			d_flag_src(d_flag_src),
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
		return detail.enactor->template Scan<AutotunedPolicy>(
			detail.d_dest,
			detail.d_src,
			detail.d_flag_src,
			detail.num_elements,
			detail.max_grid_size);
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
			return detail.enactor->template Scan<SmallPolicy>(
				detail.d_dest,
				detail.d_src,
				detail.d_flag_src,
				detail.num_elements,
				detail.max_grid_size);
		}

		// Invoke enactor with type
		return detail.enactor->template Scan<LargePolicy>(
			detail.d_dest,
			detail.d_src,
			detail.d_flag_src,
			detail.num_elements,
			detail.max_grid_size);
	}
};


/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/


/**
 * Performs a segmented scan pass
 */
template <typename Policy>
cudaError_t Enactor::EnactPass(
	typename Policy::T *d_dest,
	typename Policy::T *d_src,
	typename Policy::Flag *d_flag_src,
	util::CtaWorkDistribution<typename Policy::SizeT> &work,
	typename Policy::SizeT spine_elements)
{
	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			typedef typename Policy::Single Single;

			typename Policy::SingleKernelPtr SingleKernel = Policy::SingleKernel();

			SingleKernel<<<1, Single::THREADS, 0>>>(
				d_src, d_flag_src, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SingleKernel failed ", __FILE__, __LINE__))) break;

		} else {

			typedef typename Policy::Upsweep 		Upsweep;
			typedef typename Policy::Spine 			Spine;
			typedef typename Policy::Downsweep 		Downsweep;

			typename Policy::UpsweepKernelPtr UpsweepKernel = Policy::UpsweepKernel();
			typename Policy::SpineKernelPtr SpineKernel = Policy::SpineKernel();
			typename Policy::DownsweepKernelPtr DownsweepKernel = Policy::DownsweepKernel();

			typedef typename Spine::T 				SpineType;
			typedef typename Spine::Flag			SpineFlag;

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
				d_src, d_flag_src, (SpineType*) partial_spine(), (SpineFlag*) flag_spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor UpsweepKernel failed ", __FILE__, __LINE__))) break;

			// Spine scan
			SpineKernel<<<grid_size[1], Spine::THREADS, dynamic_smem[1]>>>(
				(SpineType*) partial_spine(), (SpineFlag*) flag_spine(), (SpineType*) partial_spine(), spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SpineKernel failed ", __FILE__, __LINE__))) break;

			// Downsweep scan from spine
			DownsweepKernel<<<grid_size[2], Downsweep::THREADS, dynamic_smem[2]>>>(
				d_src, d_flag_src, d_dest, (SpineType*) partial_spine(), work);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor DownsweepKernel failed ", __FILE__, __LINE__))) break;

		}
	} while (0);

	return retval;
}


/**
 * Enacts a segmented scan on the specified device data.
 */
template <typename Policy>
cudaError_t Enactor::Scan(
	typename Policy::T *d_dest,
	typename Policy::T *d_src,
	typename Policy::Flag *d_flag_src,
	typename Policy::SizeT num_elements,
	int max_grid_size)
{
	typedef typename Policy::Upsweep 	Upsweep;
	typedef typename Policy::Spine 		Spine;
	typedef typename Policy::Downsweep 	Downsweep;
	typedef typename Policy::Single 	Single;
	typedef typename Policy::T 			T;
	typedef typename Policy::SizeT 		SizeT;

	const int MIN_OCCUPANCY = B40C_MIN((int) Upsweep::CTA_OCCUPANCY, (int) Downsweep::CTA_OCCUPANCY);
	util::SuppressUnusedConstantWarning(MIN_OCCUPANCY);

	// Compute sweep grid size
	int sweep_grid_size = (Policy::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size) :
		OccupiedGridSize<Downsweep::SCHEDULE_GRANULARITY, MIN_OCCUPANCY>(num_elements, max_grid_size);

	if (num_elements <= Spine::TILE_ELEMENTS * 3) {
		// No need to upsweep reduce or downsweep scan if we can do it
		// with a single spine kernel in three or less sequential
		// tiles (i.e., instead of three back-to-back tiles where we would
		// do one tile per up/spine/down kernel)
		sweep_grid_size = 1;
	}

	// Compute spine elements: one element per CTA, rounded
	// up to nearest spine tile size
	int spine_elements = ((sweep_grid_size + Spine::TILE_ELEMENTS - 1) / Spine::TILE_ELEMENTS) * Spine::TILE_ELEMENTS;

	// Obtain a CTA work distribution
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
		if (retval = partial_spine.Setup<T>(spine_elements)) break;
		if (retval = flag_spine.Setup<T>(spine_elements)) break;

		// Invoke segmented scan pass
		if (retval = EnactPass<Policy>(d_dest, d_src, d_flag_src, work, spine_elements)) break;

	} while (0);

	return retval;
}


/**
 * Enacts a segmented scan operation on the specified device.
 */
template <
	typename T,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity(),
	ProbSizeGenre PROB_SIZE_GENRE,
	typename Flag,
	typename SizeT>
cudaError_t Enactor::Scan(
	T *d_dest,
	T *d_src,
	Flag *d_flag_src,
	SizeT num_elements,
	int max_grid_size)
{
	typedef Detail<T, Flag, SizeT, EXCLUSIVE, BinaryOp, Identity> Detail;
	typedef PolicyResolver<PROB_SIZE_GENRE> Resolver;

	Detail detail(this, d_dest, d_src, d_flag_src, num_elements, max_grid_size);

	return util::ArchDispatch<__B40C_CUDA_ARCH__, Resolver>::Enact(
		detail, PtxVersion());
}


/**
 * Enacts a segmented scan operation on the specified device data.
 */
template <
	typename T,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity(),
	typename Flag,
	typename SizeT>
cudaError_t Enactor::Scan(
	T *d_dest,
	T *d_src,
	Flag *d_flag_src,
	SizeT num_elements,
	int max_grid_size)
{
	return Scan<T, EXCLUSIVE, BinaryOp, Identity, UNKNOWN_SIZE>(
		d_dest, d_src, d_flag_src, num_elements, max_grid_size);
}



} // namespace segmented_scan
} // namespace b40c

