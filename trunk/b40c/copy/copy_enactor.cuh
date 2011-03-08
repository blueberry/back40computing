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
 * Base Copy Enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/enactor_base.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/copy/problem_config.cuh>
#include <b40c/copy/kernel_sweep.cuh>

namespace b40c {
namespace copy {


/******************************************************************************
 * CopyEnactor Declaration
 ******************************************************************************/

/**
 * Basic copy enactor class.
 */
template <typename DerivedEnactorType = void>
class CopyEnactor : public util::EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Dispatch type (either to self or to derived class as per CRTP -- curiously
	// recurring template pattern)
	typedef typename DispatchType<CopyEnactor, DerivedEnactorType>::Type Dispatch;
	Dispatch *dispatch;

	// A pair of counters in global device memory and a selector for
	// indexing into the pair.  If we perform workstealing passes, the
	// current counter can provide an atomic reference of progress.  One pass
	// resets the counter for the next
	size_t*	d_work_progress;
	int 	progress_selector;


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Performs any lazy initialization work needed for this problem type
	 */
	template <typename ProblemConfig>
	cudaError_t Setup(int sweep_grid_size);

    /**
	 * Performs a copy pass
	 */
	template <typename ProblemConfig>
	cudaError_t CopyPass(
		typename ProblemConfig::Sweep::T *d_dest,
		typename ProblemConfig::Sweep::T *d_src,
		util::CtaWorkDistribution<typename ProblemConfig::Sweep::SizeT> &work,
		int extra_bytes);

public:

	/**
	 * Constructor
	 */
	CopyEnactor();


	/**
     * Destructor
     */
    virtual ~CopyEnactor();


	/**
	 * Enacts a copy on the specified device data.
	 *
	 * For generating copy kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
	 *
	 * @param d_dest
	 * 		Pointer to array of elements to be copyd
	 * @param d_src
	 * 		Pointer to result location
	 * @param num_elements
	 * 		Number of elements to copy
	 * @param max_grid_size
	 * 		Optional upper-bound on the number of CTAs to launch.
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename ProblemConfig>
	cudaError_t Enact(
		typename ProblemConfig::Sweep::T *d_dest,
		typename ProblemConfig::Sweep::T *d_src,
		typename ProblemConfig::Sweep::SizeT num_elements,
		int extra_bytes = 0,
		int max_grid_size = 0);
};



/******************************************************************************
 * CopyEnactor Implementation
 ******************************************************************************/

/**
 * Performs any lazy initialization work needed for this problem type
 */
template <typename DerivedEnactorType>
template <typename ProblemConfig>
cudaError_t CopyEnactor<DerivedEnactorType>::Setup(int sweep_grid_size)
{
	cudaError_t retval = cudaSuccess;
	do {
		// Optional setup for workstealing passes
		if (ProblemConfig::Sweep::WORK_STEALING) {

			// Make sure that our progress counters are allocated
			if (d_work_progress == NULL) {
				// Allocate
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_work_progress, sizeof(size_t) * 2),
					"CopyEnactor cudaMalloc d_work_progress failed", __FILE__, __LINE__)) break;

				// Initialize
				size_t h_work_progress[2] = {0, 0};
				if (retval = util::B40CPerror(cudaMemcpy(d_work_progress, h_work_progress, sizeof(size_t) * 2, cudaMemcpyHostToDevice),
					"CopyEnactor cudaMemcpy d_work_progress failed", __FILE__, __LINE__)) break;
			}

			// Update our progress counter selector to index the next progress counter
			progress_selector ^= 1;
		}
	} while (0);

	return retval;
}



/**
 * Performs a copy pass
 */
template <typename DerivedEnactorType>
template <typename ProblemConfig>
cudaError_t CopyEnactor<DerivedEnactorType>::CopyPass(
	typename ProblemConfig::Sweep::T *d_dest,
	typename ProblemConfig::Sweep::T *d_src,
	util::CtaWorkDistribution<typename ProblemConfig::Sweep::SizeT> &work,
	int extra_bytes)
{
	typedef typename ProblemConfig::Sweep::T T;

	cudaError_t retval = cudaSuccess;
	int dynamic_smem = 0;

	// Sweep copy
	SweepCopyKernel<typename ProblemConfig::Sweep>
			<<<work.grid_size, ProblemConfig::Sweep::THREADS, dynamic_smem>>>(
		d_src, d_dest, d_work_progress, work, progress_selector, extra_bytes);
	if (DEBUG) retval = util::B40CPerror(cudaThreadSynchronize(), "CopyEnactor SweepCopyKernel failed ", __FILE__, __LINE__);

	return retval;
}


/**
 * Constructor
 */
template <typename DerivedEnactorType>
CopyEnactor<DerivedEnactorType>::CopyEnactor() :
	d_work_progress(NULL),
	progress_selector(0)
{
	dispatch = static_cast<Dispatch*>(this);
}


/**
 * Destructor
 */
template <typename DerivedEnactorType>
CopyEnactor<DerivedEnactorType>::~CopyEnactor()
{
	if (d_work_progress) {
		util::B40CPerror(cudaFree(d_work_progress), "CopyEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
	}
}

    
/**
 * Enacts a copy on the specified device data.
 */
template <typename DerivedEnactorType>
template <typename ProblemConfig>
cudaError_t CopyEnactor<DerivedEnactorType>::Enact(
	typename ProblemConfig::Sweep::T *d_dest,
	typename ProblemConfig::Sweep::T *d_src,
	typename ProblemConfig::Sweep::SizeT num_elements,
	int extra_bytes,
	int max_grid_size)
{
	typedef typename ProblemConfig::Sweep Sweep;
	typedef typename Sweep::T T;
	typedef typename Sweep::SizeT SizeT;

	// Compute sweep grid size
	int sweep_grid_size = (ProblemConfig::OVERSUBSCRIBED_GRID_SIZE) ?
				OversubscribedGridSize<Sweep::SCHEDULE_GRANULARITY, Sweep::CTA_OCCUPANCY>(num_elements, max_grid_size) :
				OccupiedGridSize<Sweep::SCHEDULE_GRANULARITY, Sweep::CTA_OCCUPANCY>(num_elements, max_grid_size);

	// Obtain a CTA work distribution for copying items of type T
	util::CtaWorkDistribution<SizeT> work(num_elements, Sweep::SCHEDULE_GRANULARITY, sweep_grid_size);

	if (DEBUG) {
		printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n",
			cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
		printf("Sweep: \t\t[sweep_grid_size: %d, threads %d, SizeT %lu bytes, workstealing: %s, tile_elements: %d]\n",
			work.grid_size, Sweep::THREADS, (unsigned long) sizeof(SizeT), Sweep::WORK_STEALING ? "true" : "false", Sweep::TILE_ELEMENTS);
		printf("Work: \t\t[element bytes: %lu, num_elements: %lu, schedule_granularity: %d, total_grains: %lu, grains_per_cta: %lu extra_grains: %lu]\n",
			(unsigned long) sizeof(T), (unsigned long) work.num_elements, Sweep::SCHEDULE_GRANULARITY, (unsigned long) work.total_grains, (unsigned long) work.grains_per_cta, (unsigned long) work.extra_grains);
	}

	cudaError_t retval = cudaSuccess;
	do {
		// Perform any lazy initialization work
		if (retval = Setup<ProblemConfig>(sweep_grid_size)) break;

		// Invoke copy kernel
		if (retval = dispatch->template CopyPass<ProblemConfig>(d_dest, d_src, work, extra_bytes)) break;

	} while (0);

	// Cleanup
	if (retval) {
		// We had an error, which means that the device counters may not be
		// properly initialized for the next pass: reset them.
		if (d_work_progress) {
			util::B40CPerror(cudaFree(d_work_progress), "CopyEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
			d_work_progress = NULL;
		}
	}

	return retval;
}


}// namespace copy
}// namespace b40c

