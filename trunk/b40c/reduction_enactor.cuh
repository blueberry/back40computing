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

#include <b40c/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/reduction/granularity.cuh>
#include <b40c/reduction/kernel_upsweep.cuh>
#include <b40c/reduction/kernel_spine.cuh>

namespace b40c {


/******************************************************************************
 * ReductionEnactor Declaration
 ******************************************************************************/

/**
 * Basic reduction enactor class.
 */
template <typename DerivedEnactorType = void>
class ReductionEnactor : public EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Dispatch type (either to self or to derived class as per CRTP -- curiously
	// recurring template pattern)
	typedef typename DispatchType<ReductionEnactor, DerivedEnactorType>::Type Dispatch;
	Dispatch *dispatch;

	// A pair of counters in global device memory and a selector for
	// indexing into the pair.  If we perform workstealing passes, the
	// current counter can provide an atomic reference of progress.  One pass
	// resets the counter for the next
	size_t*	d_work_progress;
	int 	progress_selector;

	// Temporary device storage needed for reducing partials produced
	// by separate CTAs
	void *d_spine;

	// Number of bytes backed by d_spine
	int spine_bytes;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Performs any lazy initialization work needed for this problem type
	 */
	template <typename ReductionConfig>
	cudaError_t Setup(int sweep_grid_size, int spine_elements);

    /**
	 * Performs a reduction pass
	 */
	template <typename ReductionConfig>
	cudaError_t ReductionPass(
		typename ReductionConfig::Upsweep::T *d_dest,
		typename ReductionConfig::Upsweep::T *d_src,
		util::CtaWorkDistribution<typename ReductionConfig::Upsweep::SizeT> &work,
		int spine_elements);

public:

	/**
	 * Constructor
	 */
	ReductionEnactor();


	/**
     * Destructor
     */
    virtual ~ReductionEnactor();


	/**
	 * Enacts a reduction on the specified device data.
	 *
	 * For generating reduction kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be reduced
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to reduce
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename ReductionConfig>
	cudaError_t Enact(
		typename ReductionConfig::Upsweep::T *d_dest,
		typename ReductionConfig::Upsweep::T *d_src,
		typename ReductionConfig::Upsweep::SizeT num_elements,
		int max_grid_size = 0);
};



/******************************************************************************
 * ReductionEnactor Implementation
 ******************************************************************************/

/**
 * Performs any lazy initialization work needed for this problem type
 */
template <typename DerivedEnactorType>
template <typename ReductionConfig>
cudaError_t ReductionEnactor<DerivedEnactorType>::Setup(int sweep_grid_size, int spine_elements)
{
	cudaError_t retval = cudaSuccess;
	do {
		// Make sure our spine is big enough
		int problem_spine_bytes = spine_elements * sizeof(typename ReductionConfig::Upsweep::T);
		if (problem_spine_bytes > spine_bytes) {
			if (d_spine) {
				if (retval = util::B40CPerror(cudaFree(d_spine),
					"ReductionEnactor cudaFree d_spine failed", __FILE__, __LINE__)) break;
			}

			spine_bytes = problem_spine_bytes;

			if (retval = util::B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
				"ReductionEnactor cudaMalloc d_spine failed", __FILE__, __LINE__)) break;
		}

		// Optional setup for workstealing passes
		if (ReductionConfig::Upsweep::WORK_STEALING) {

			// Make sure that our progress counters are allocated
			if (d_work_progress == NULL) {
				// Allocate
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_work_progress, sizeof(size_t) * 2),
					"ReductionEnactor cudaMalloc d_work_progress failed", __FILE__, __LINE__)) break;

				// Initialize
				size_t h_work_progress[2] = {0, 0};
				if (retval = util::B40CPerror(cudaMemcpy(d_work_progress, h_work_progress, sizeof(size_t) * 2, cudaMemcpyHostToDevice),
					"ReductionEnactor cudaMemcpy d_work_progress failed", __FILE__, __LINE__)) break;
			}

			// Update our progress counter selector to index the next progress counter
			progress_selector ^= 1;
		}
	} while (0);

	return retval;
}



/**
 * Performs a reduction pass
 */
template <typename DerivedEnactorType>
template <typename ReductionConfig>
cudaError_t ReductionEnactor<DerivedEnactorType>::ReductionPass(
	typename ReductionConfig::Upsweep::T *d_dest,
	typename ReductionConfig::Upsweep::T *d_src,
	util::CtaWorkDistribution<typename ReductionConfig::Upsweep::SizeT> &work,
	int spine_elements)
{
	using namespace reduction;

	typedef typename ReductionConfig::Upsweep::T T;

	cudaError_t retval = cudaSuccess;
	do {
		if (work.grid_size == 1) {

			// No need to upsweep reduce if there's only one CTA in the upsweep grid
			int dynamic_smem = 0;
			SpineReductionKernel<typename ReductionConfig::Spine>
					<<<work.grid_size, ReductionConfig::Spine::THREADS, dynamic_smem>>>(
				d_src, d_dest, work.num_elements);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor SpineReductionKernel failed ", __FILE__, __LINE__))) break;

		} else {

			int dynamic_smem[2] = 	{0, 0};
			int grid_size[2] = 		{work.grid_size, 1};

			// Tuning option for dynamic smem allocation
			if (ReductionConfig::UNIFORM_SMEM_ALLOCATION) {

				// We need to compute dynamic smem allocations to ensure all three
				// kernels end up allocating the same amount of smem per CTA

				// Get kernel attributes
				cudaFuncAttributes upsweep_kernel_attrs, spine_kernel_attrs;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&upsweep_kernel_attrs, UpsweepReductionKernel<typename ReductionConfig::Upsweep>),
					"ReductionEnactor cudaFuncGetAttributes upsweep_kernel_attrs failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaFuncGetAttributes(&spine_kernel_attrs, SpineReductionKernel<typename ReductionConfig::Spine>),
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
			UpsweepReductionKernel<typename ReductionConfig::Upsweep>
					<<<grid_size[0], ReductionConfig::Upsweep::THREADS, dynamic_smem[0]>>>(
				d_src, (T*) d_spine, d_work_progress, work, progress_selector);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor UpsweepReductionKernel failed ", __FILE__, __LINE__))) break;

			// Spine reduction
			SpineReductionKernel<typename ReductionConfig::Spine>
					<<<grid_size[1], ReductionConfig::Spine::THREADS, dynamic_smem[1]>>>(
				(T*) d_spine, d_dest, spine_elements);
			if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "ReductionEnactor SpineReductionKernel failed ", __FILE__, __LINE__))) break;
		}
	} while (0);

	return retval;
}


/**
 * Constructor
 */
template <typename DerivedEnactorType>
ReductionEnactor<DerivedEnactorType>::ReductionEnactor() :
	d_work_progress(NULL),
	progress_selector(0),
	d_spine(NULL),
	spine_bytes(0)
{
	dispatch = static_cast<Dispatch*>(this);
}


/**
 * Destructor
 */
template <typename DerivedEnactorType>
ReductionEnactor<DerivedEnactorType>::~ReductionEnactor()
{
	if (d_work_progress) {
		util::B40CPerror(cudaFree(d_work_progress), "ReductionEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
	}
	if (d_spine) {
		util::B40CPerror(cudaFree(d_spine), "ReductionEnactor cudaFree d_spine failed: ", __FILE__, __LINE__);
	}
}

    
/**
 * Enacts a reduction on the specified device data.
 */
template <typename DerivedEnactorType>
template <typename ReductionConfig>
cudaError_t ReductionEnactor<DerivedEnactorType>::Enact(
	typename ReductionConfig::Upsweep::T *d_dest,
	typename ReductionConfig::Upsweep::T *d_src,
	typename ReductionConfig::Upsweep::SizeT num_elements,
	int max_grid_size)
{
	typedef typename ReductionConfig::Upsweep Upsweep;
	typedef typename ReductionConfig::Spine Spine;
	typedef typename Upsweep::T T;
	typedef typename Upsweep::SizeT SizeT;

	// Compute sweep grid size
	int sweep_grid_size = (num_elements <= Spine::TILE_ELEMENTS) ?
		1 :
		(ReductionConfig::OVERSUBSCRIBED_GRID_SIZE) ?
				OversubscribedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size) :
				OccupiedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size);

	// Compute spine elements (round up to nearest spine tile elements)
	int spine_elements = sweep_grid_size;

	// Obtain a CTA work distribution for copying items of type T
	util::CtaWorkDistribution<SizeT> work(num_elements, Upsweep::SCHEDULE_GRANULARITY, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
		if (sweep_grid_size > 1) {
			printf("Upsweep: \t[sweep_grid_size: %d, threads %d, SizeT %d bytes, workstealing: %s, tile_elements: %d]\n",
				work.grid_size, Upsweep::THREADS, sizeof(SizeT), Upsweep::WORK_STEALING ? "true" : "false", Upsweep::TILE_ELEMENTS);
			printf("Spine: \t\t[threads: %d, spine_elements: %d, tile_elements: %d]\n",
				Spine::THREADS, spine_elements, Spine::TILE_ELEMENTS);
			printf("Work: \t\t[element bytes: %d, num_elements: %d, schedule_granularity: %d, total_grains: %d, grains_per_cta: %d, extra_grains: %d]\n",
				sizeof(T), work.num_elements, Upsweep::SCHEDULE_GRANULARITY, work.total_grains, work.grains_per_cta, work.extra_grains);
		} else {
			printf("Spine: \t\t[threads: %d, tile_elements: %d]\n",
				Spine::THREADS, Spine::TILE_ELEMENTS);
		}
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Perform any lazy initialization work
		if (retval = Setup<ReductionConfig>(sweep_grid_size, spine_elements)) break;

		// Invoke reduction kernel
		if (retval = dispatch->template ReductionPass<ReductionConfig>(d_dest, d_src, work, spine_elements)) break;

	} while (0);

	// Cleanup
	if (retval) {
		// We had an error, which means that the device counters may not be
		// properly initialized for the next pass: reset them.
		if (d_work_progress) {
			util::B40CPerror(cudaFree(d_work_progress), "ReductionEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
			d_work_progress = NULL;
		}
	}

	return retval;
}



}// namespace b40c

