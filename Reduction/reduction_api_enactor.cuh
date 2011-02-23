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

#include <stdio.h>

#include "b40c_kernel_utils.cuh"
#include "b40c_enactor_base.cuh"
#include "reduction_kernel.cuh"

namespace b40c {
using namespace reduction;


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
		CtaWorkDistribution<typename ReductionConfig::Upsweep::SizeT> &work,
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
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename ReductionConfig>
	cudaError_t Enact(
		typename ReductionConfig::Upsweep::T *d_dest,
		typename ReductionConfig::Upsweep::T *d_src,
		size_t num_elements,
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
				if (retval = B40CPerror(cudaFree(d_spine),
					"LsbSortEnactor cudaFree d_spine failed", __FILE__, __LINE__)) break;
			}

			spine_bytes = problem_spine_bytes;

			if (retval = B40CPerror(cudaMalloc((void**) &d_spine, spine_bytes),
				"LsbSortEnactor cudaMalloc d_spine failed", __FILE__, __LINE__)) break;
		}

		// Optional setup for workstealing passes
		if (ReductionConfig::Upsweep::WORK_STEALING) {

			// Make sure that our progress counters are allocated
			if (d_work_progress == NULL) {
				// Allocate
				if (retval = B40CPerror(cudaMalloc((void**) &d_work_progress, sizeof(size_t) * 2),
					"LsbSortEnactor cudaMalloc d_work_progress failed", __FILE__, __LINE__)) break;

				// Initialize
				size_t h_work_progress[2] = {0, 0};
				if (retval = B40CPerror(cudaMemcpy(d_work_progress, h_work_progress, sizeof(size_t) * 2, cudaMemcpyHostToDevice),
					"LsbSortEnactor cudaMemcpy d_work_progress failed", __FILE__, __LINE__)) break;
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
	CtaWorkDistribution<typename ReductionConfig::Upsweep::SizeT> &work,
	int spine_elements)
{
	typedef typename ReductionConfig::Upsweep::T T;

	int dynamic_smem = 0;
	cudaError_t retval = cudaSuccess;

	if (work.grid_size == 1) {

		// No need to scan the spine if there's only one CTA in the upsweep grid
		UpsweepReductionKernel<ReductionConfig::Upsweep>
				<<<work.grid_size, ReductionConfig::Upsweep::THREADS, dynamic_smem>>>(
			d_src, d_dest, d_work_progress, work, progress_selector);

	} else {

		// Upsweep reduction into spine
		UpsweepReductionKernel<ReductionConfig::Upsweep>
				<<<work.grid_size, ReductionConfig::Upsweep::THREADS, dynamic_smem>>>(
			d_src, (T*) d_spine, d_work_progress, work, progress_selector);

		// Spine reduction
		SpineReductionKernel<ReductionConfig::Spine>
				<<<1, ReductionConfig::Spine::THREADS, dynamic_smem>>>(
			(T*) d_spine, d_dest, spine_elements);
	}

	if (DEBUG) {
		retval = B40CPerror(cudaThreadSynchronize(), "ReductionEnactor:: ReductionKernel failed ", __FILE__, __LINE__);
	}

	return retval;
}


/**
 * Constructor
 */
template <typename DerivedEnactorType>
ReductionEnactor<DerivedEnactorType>::ReductionEnactor() :
	d_work_progress(NULL),
	progress_selector(0),
	d_spine(NULL)
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
		B40CPerror(cudaFree(d_work_progress), "ReductionEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
	}
	if (d_spine) {
		B40CPerror(cudaFree(d_spine), "ReductionEnactor cudaFree d_spine failed: ", __FILE__, __LINE__);
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
	size_t num_elements,
	int max_grid_size)
{
	typedef typename ReductionConfig::Upsweep Upsweep;
	typedef typename Upsweep::T T;
	typedef typename Upsweep::SizeT SizeT;

	int sweep_grid_size = (Upsweep::WORK_STEALING) ?
		OccupiedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size) :
		OversubscribedGridSize<Upsweep::SCHEDULE_GRANULARITY, Upsweep::CTA_OCCUPANCY>(num_elements, max_grid_size);

	int spine_elements = sweep_grid_size;

	// Obtain a CTA work distribution for copying items of type T
	CtaWorkDistribution<SizeT> work(num_elements, Upsweep::SCHEDULE_GRANULARITY, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
		printf("Upsweep: \t[sweep_grid_size: %d, threads %d, SizeT %d bytes, workstealing: %s]\n",
			work.grid_size, Upsweep::THREADS, sizeof(typename Upsweep::SizeT), Upsweep::WORK_STEALING ? "true" : "false");
		if (sweep_grid_size > 1) printf("Spine: \t\t[threads: %d, spine_elements: %d]\n",
			ReductionConfig::Spine::THREADS, spine_elements);
		printf("Work: \t\t[element bytes: %d, num_elements: %d, schedule_granularity: %d, total_grains: %d, grains_per_cta: %d, extra_grains: %d]\n",
			sizeof(typename Upsweep::T), work.num_elements, Upsweep::SCHEDULE_GRANULARITY, work.total_grains, work.grains_per_cta, work.extra_grains);
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
			B40CPerror(cudaFree(d_work_progress), "ReductionEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
			d_work_progress = NULL;
		}
	}

	return retval;
}



}// namespace b40c

