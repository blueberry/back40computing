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
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/copy/sweep_kernel.cuh>

namespace b40c {
namespace copy {


/******************************************************************************
 * Enactor Declaration
 ******************************************************************************/

/**
 * Basic copy enactor class.
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


	//-----------------------------------------------------------------------------
	// Utility Routines
	//-----------------------------------------------------------------------------

	/**
	 * Performs any lazy per-pass initialization work needed for this problem type
	 */
	template <typename ProblemConfig>
	cudaError_t Setup(int sweep_grid_size);

    /**
	 * Performs a copy pass
	 */
	template <typename ProblemConfig>
	cudaError_t EnactPass(
		typename ProblemConfig::Sweep::T *d_dest,
		typename ProblemConfig::Sweep::T *d_src,
		util::CtaWorkDistribution<typename ProblemConfig::Sweep::SizeT> &work,
		int extra_bytes);

	/**
	 * Enacts a copy on the specified device data.
	 */
	template <typename ProblemConfig, typename EnactorType>
	cudaError_t EnactInternal(
		typename ProblemConfig::Sweep::T *d_dest,
		typename ProblemConfig::Sweep::T *d_src,
		typename ProblemConfig::Sweep::SizeT num_elements,
		int extra_bytes,
		int max_grid_size);	
	
public:

	/**
	 * Constructor
	 */
	Enactor() {}


	/**
	 * Enacts a copy on the specified device data using the specified 
	 * granularity configuration
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
		int max_grid_size = 0);
};



/******************************************************************************
 * Enactor Implementation
 ******************************************************************************/

/**
 * Performs any lazy initialization work needed for this problem type
 */
template <typename ProblemConfig>
cudaError_t Enactor::Setup(int sweep_grid_size)
{
	// If we're work-stealing, make sure our work progress is set up
	// for the next pass
	if (ProblemConfig::Sweep::WORK_STEALING) {
		return work_progress.Setup();
	} else {
		return cudaSuccess;
	}
}


/**
 * Performs a copy pass
 */
template <typename ProblemConfig>
cudaError_t Enactor::EnactPass(
	typename ProblemConfig::Sweep::T *d_dest,
	typename ProblemConfig::Sweep::T *d_src,
	util::CtaWorkDistribution<typename ProblemConfig::Sweep::SizeT> &work,
	int extra_bytes)
{
	typedef typename ProblemConfig::Sweep::T T;

	cudaError_t retval = cudaSuccess;
	int dynamic_smem = 0;

	// Sweep copy
	SweepKernel<typename ProblemConfig::Sweep>
			<<<work.grid_size, ProblemConfig::Sweep::THREADS, dynamic_smem>>>(
		d_src, d_dest, work, work_progress, extra_bytes);
	if (DEBUG) retval = util::B40CPerror(cudaThreadSynchronize(), "Enactor SweepCopyKernel failed ", __FILE__, __LINE__);

	return retval;
}

    
/**
 * Enacts a copy on the specified device data.
 */
template <typename ProblemConfig, typename EnactorType>
cudaError_t Enactor::EnactInternal(
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
	util::CtaWorkDistribution<SizeT> work;
	work.template Init<Sweep::LOG_SCHEDULE_GRANULARITY>(num_elements, sweep_grid_size);

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

		// Invoke copy pass
		EnactorType *dipatch = static_cast<EnactorType *>(this);
		if (retval = dipatch->template EnactPass<ProblemConfig>(
			d_dest, d_src, work, extra_bytes)) break;

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
 * Enacts a copy on the specified device data.
 */
template <typename ProblemConfig>
cudaError_t Enactor::Enact(
	typename ProblemConfig::Sweep::T *d_dest,
	typename ProblemConfig::Sweep::T *d_src,
	typename ProblemConfig::Sweep::SizeT num_elements,
	int max_grid_size)
{
	return EnactInternal<ProblemConfig, Enactor>(
		d_dest, d_src, num_elements, 0, max_grid_size);
}


}// namespace copy
}// namespace b40c

