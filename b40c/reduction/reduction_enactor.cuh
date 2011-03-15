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
 * Base Reduction Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/spine.cuh>
#include <b40c/util/work_progress.cuh>
#include <b40c/reduction/problem_config.cuh>
#include <b40c/reduction/kernel_upsweep.cuh>
#include <b40c/reduction/kernel_spine.cuh>

namespace b40c {
namespace reduction {


/******************************************************************************
 * ReductionEnactor Declaration
 ******************************************************************************/

/**
 * Basic reduction enactor class.
 */
class ReductionEnactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Temporary device storage needed for managing work-stealing progress
	// within a kernel invocation.
	util::WorkProgressLifetime work_progress;

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	util::Spine spine;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Performs any lazy per-pass initialization work needed for this problem type
	 */
	template <typename ProblemConfig>
	cudaError_t Setup(int sweep_grid_size, int spine_elements);

    /**
	 * Performs a reduction pass
	 */
	template <typename ProblemConfig>
	cudaError_t ReductionPass(
		typename ProblemConfig::T *d_dest,
		typename ProblemConfig::T *d_src,
		util::CtaWorkDistribution<typename ProblemConfig::SizeT> &work,
		int spine_elements);

	/**
	 * Enacts a reduction on the specified device data.
	 */
	template <typename ProblemConfig, typename EnactorType>
	cudaError_t EnactInternal(
		typename ProblemConfig::T *d_dest,
		typename ProblemConfig::T *d_src,
		typename ProblemConfig::SizeT num_elements,
		int max_grid_size);

public:

	/**
	 * Constructor
	 */
	ReductionEnactor() {}


	/**
	 * Enacts a reduction on the specified device data.
	 *
	 * For generating reduction kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
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
	template <typename ProblemConfig>
	cudaError_t Enact(
		typename ProblemConfig::T *d_dest,
		typename ProblemConfig::T *d_src,
		typename ProblemConfig::SizeT num_elements,
		int max_grid_size = 0);
};



/******************************************************************************
 * ReductionEnactor Implementation
 ******************************************************************************/

/**
 * Performs any lazy initialization work needed for this problem type
 */
template <typename ProblemConfig>
cudaError_t ReductionEnactor::Setup(int sweep_grid_size, int spine_elements)
{
	typedef typename ProblemConfig::T T;

	cudaError_t retval = cudaSuccess;
	do {
		// Make sure our spine is big enough
		if (retval = spine.Setup<T>(sweep_grid_size, spine_elements)) break;

		// If we're work-stealing, make sure our work progress is set up
		// for the next pass
		if (ProblemConfig::Upsweep::WORK_STEALING) {
			if (retval = work_progress.Setup()) break;
		}
	} while (0);

	return retval;
}



/**
 * Performs a reduction pass
 */
template <typename ProblemConfig>
cudaError_t ReductionEnactor::ReductionPass(
	typename ProblemConfig::T *d_dest,
	typename ProblemConfig::T *d_src,
	util::CtaWorkDistribution<typename ProblemConfig::SizeT> &work,
	int spine_elements)
{
	typedef typename ProblemConfig::Upsweep Upsweep;
	typedef typename ProblemConfig::Spine Spine;
	typedef typename Upsweep::T T;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			SpineReductionKernel<Spine><<<1, Spine::THREADS, 0>>>(
				d_src, d_dest, work.num_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor SpineReductionKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[2] = 	{0, 0};
			int grid_size[2] = 		{work.grid_size, 1};

			// Tuning option for dynamic smem allocation
			if (ProblemConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepReductionKernel<typename ProblemConfig::Upsweep>),
					"ReductionEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineReductionKernel<typename ProblemConfig::Spine>),
					"ReductionEnactor cudaFuncGetAttributes spine_kernel_attrs failed", __FILE__, __LINE__)) break;

				int max_static_smem = B40C_MAX(upsweep_kernel_attrs.sharedSizeBytes, spine_kernel_attrs.sharedSizeBytes);

				dynamic_smem[0] = max_static_smem - upsweep_kernel_attrs.sharedSizeBytes;
				dynamic_smem[1] = max_static_smem - spine_kernel_attrs.sharedSizeBytes;
			}

			// Tuning option for spine-scan kernel grid size
			if (ProblemConfig::UNIFORM_GRID_SIZE) {
				grid_size[1] = grid_size[0]; 				// We need to make sure that all kernels launch the same number of CTAs
			}

			// Upsweep reduction into spine
			UpsweepReductionKernel<typename ProblemConfig::Upsweep>
					<<<grid_size[0], ProblemConfig::Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) spine(), work, work_progress);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

			// Spine reduction
			SpineReductionKernel<typename ProblemConfig::Spine>
					<<<grid_size[1], ProblemConfig::Spine::THREADS, dynamic_smem[1]>>>(
				(T*) spine(), d_dest, spine_elements);

			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor SpineReductionKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Enacts a reduction on the specified device data.
 */
template <typename ProblemConfig, typename EnactorType>
cudaError_t ReductionEnactor::EnactInternal(
	typename ProblemConfig::T *d_dest,
	typename ProblemConfig::T *d_src,
	typename ProblemConfig::SizeT num_elements,
	int max_grid_size)
{
	typedef typename ProblemConfig::Upsweep Upsweep;
	typedef typename ProblemConfig::Spine Spine;
	typedef typename Upsweep::T T;
	typedef typename Upsweep::SizeT SizeT;

	// Compute sweep grid size
	int sweep_grid_size = (ProblemConfig::OVERSUBSCRIBED_GRID_SIZE) ?
		OversubscribedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size) :
		OccupiedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size);

	if (num_elements <= Spine::TILE_ELEMENTS * 2) {
		// No need to upsweep reduce if we can do it with a single spine
		// kernel in two or less sequential tiles (i.e., instead of two
		// back-to-back tiles where we would one tile per up/spine kernel)
		sweep_grid_size = 1;
	}

	// Compute spine elements (round up to nearest spine tile elements)
	int spine_elements = sweep_grid_size;

	// Obtain a CTA work distribution for copying items of type T
	util::CtaWorkDistribution<SizeT> work(num_elements, Upsweep::SCHEDULE_GRANULARITY, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
		if (sweep_grid_size > 1) {
			printf("Upsweep: \t[sweep_grid_size: %d, threads %d, SizeT %lu bytes, workstealing: %s, tile_elements: %d]\n",
				work.grid_size, Upsweep::THREADS, (unsigned long) sizeof(SizeT), Upsweep::WORK_STEALING ? "true" : "false", Upsweep::TILE_ELEMENTS);
			printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
				Spine::THREADS, spine_elements, Spine::TILE_ELEMENTS);
			printf("Work: \t\t[element bytes: %lu, num_elements: %lu, schedule_granularity: %d, total_grains: %lu, grains_per_cta: %lu extra_grains: %lu]\n",
				(unsigned long) sizeof(T), (unsigned long) work.num_elements, Upsweep::SCHEDULE_GRANULARITY, (unsigned long) work.total_grains, (unsigned long) work.grains_per_cta, (unsigned long) work.extra_grains);
		} else {
			printf("Spine: \t\t[threads: %d, tile_elements: %d]\n",
				Spine::THREADS, Spine::TILE_ELEMENTS);
		}
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Perform any lazy initialization work
		if (retval = Setup<ProblemConfig>(sweep_grid_size, spine_elements)) break;

		// Invoke reduction pass
		EnactorType *dipatch = static_cast<EnactorType *>(this);
		if (retval = dipatch->template ReductionPass<ProblemConfig>(
			d_dest, d_src, work, spine_elements)) break;

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
template <typename ProblemConfig>
cudaError_t ReductionEnactor::Enact(
	typename ProblemConfig::T *d_dest,
	typename ProblemConfig::T *d_src,
	typename ProblemConfig::SizeT num_elements,
	int max_grid_size)
{
	return EnactInternal<ProblemConfig, ReductionEnactor>(d_dest, d_src, num_elements, max_grid_size);
}


}// namespace reduction
}// namespace b40c

