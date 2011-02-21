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
 * Base Memcopy Enactor
 ******************************************************************************/

#pragma once

#include <stdio.h>

#include "b40c_kernel_utils.cuh"
#include "b40c_enactor_base.cuh"
#include "memcopy_kernel.cuh"

namespace b40c {
using namespace memcopy;


/**
 * Basic memcopy enactor class.
 */
template <typename DerivedEnactorType = void>
class MemcopyEnactor : public EnactorBase
{
protected:

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Dispatch type (to self, or to derived class as per CRTP -- curiously
	// recurring template pattern
	typedef typename DispatchType<MemcopyEnactor, DerivedEnactorType>::Type Dispatch;

	// A pair of counters in global device memory and a selector for
	// indexing into the pair.  If we perform workstealing passes, the
	// current counter can provide an atomic reference of progress.  One pass
	// resets the counter for the next
	size_t*	d_work_progress;
	int 	progress_selector;


	//-----------------------------------------------------------------------------
	// Memcopy Pass
	//-----------------------------------------------------------------------------

    /**
	 * Performs a distribution sorting pass over a single digit place
	 */
	template <typename MemcopyConfig>
	cudaError_t MemcopyPass(
		typename MemcopyConfig::T *d_dest,
		typename MemcopyConfig::T *d_src,
		CtaWorkDistribution<typename MemcopyConfig::SizeT> &work,
		int sweep_grid_size)
	{
		typedef MemcopyKernelConfig<MemcopyConfig> MemcopyKernelConfigType;
		int dynamic_smem = 0;

		cudaError_t retval = cudaSuccess;
		do {

			MemcopyKernel<MemcopyKernelConfigType><<<sweep_grid_size, MemcopyKernelConfigType::THREADS, dynamic_smem>>>(
				d_dest, d_src, d_work_progress, work, progress_selector, 0);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(),
				"MemcopyEnactor:: MemcopyKernel failed ", __FILE__, __LINE__))) break;

		} while (0);

		return retval;
	}


public:

	/**
	 * Constructor.
	 */
	MemcopyEnactor() : d_work_progress(NULL), progress_selector(0) {}


	/**
     * Destructor
     */
    virtual ~MemcopyEnactor()
    {
   		if (d_work_progress) {
   			B40CPerror(cudaFree(d_work_progress), "MemcopyEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
   		}
    }

    
	/**
	 * Enacts a memcopy on the specified device data.
	 *
	 * For generating memcopy kernels having computational granularities in accordance
	 * with user-supplied granularity-specialization types.  (Useful for auto-tuning.)
	 *
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	template <typename MemcopyConfig>
	cudaError_t Enact(
		typename MemcopyConfig::T *d_dest,
		typename MemcopyConfig::T *d_src,
		size_t length,
		int max_grid_size = 0)
	{
		typedef typename MemcopyConfig::T T;
		typedef typename MemcopyConfig::SizeT SizeT;

		const int SCHEDULE_GRANULARITY 	= 1 << MemcopyConfig::LOG_SCHEDULE_GRANULARITY;

		int sweep_grid_size = SweepGridSize<SCHEDULE_GRANULARITY, MemcopyConfig::CTA_OCCUPANCY>(
			length, max_grid_size);
		
		CtaWorkDistribution<SizeT> work(length, SCHEDULE_GRANULARITY, sweep_grid_size);

		if (DEBUG) {
			printf("\n\n");
			printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n", 
				cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
			printf("Memcopy: \t[grid_size: %d, threads %d, SizeT %d bytes]\n",
				sweep_grid_size, 1 << MemcopyConfig::LOG_THREADS, sizeof(SizeT));
			printf("Work: \t\t[num_elements: %d, schedule_granularity: %d, total_grains: %d, grains_per_cta: %d, extra_grains: %d]\n",
				work.num_elements, SCHEDULE_GRANULARITY, work.total_grains, work.grains_per_cta, work.extra_grains);
			printf("\n\n");
		}
		
		cudaError_t retval = cudaSuccess;
		do {

			// Work-stealing setup
			if (MemcopyConfig::WORK_STEALING) {

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

			// Invoke memcopy kernel
			if (retval = ((Dispatch *) this)->template MemcopyPass<MemcopyConfig>(d_dest, d_src, work, sweep_grid_size)) break;

		} while (0);

		if (retval) {
			// We had an error, which means that the device counters may not be
			// properly initialized for the next pass: reset them.
	   		if (d_work_progress) {
	   			B40CPerror(cudaFree(d_work_progress), "MemcopyEnactor cudaFree d_work_progress failed: ", __FILE__, __LINE__);
	   			d_work_progress = NULL;
	   		}
		}

	    return retval;
	}
};



}// namespace b40c

