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
#include "memcopy_kernel.cuh"

namespace b40c {
using namespace memcopy;


/**
 * Basic mem-copy enactor class.
 */
template <typename DerivedEnactorType = void>
class MemcopyEnactor
{
protected:

	//---------------------------------------------------------------------
	// Specialize templated dispatch to self (no derived class) or 
	// derived class (CRTP -- curiously recurring template pattern)
	//---------------------------------------------------------------------

	template <typename DerivedType, int __dummy = 0>
	struct DispatchType 
	{
		typedef DerivedType Type;
	};
	
	template <int __dummy>
	struct DispatchType<void, __dummy>
	{
		typedef MemcopyEnactor<void> Type;
	};

	
	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------
		
	// Device properties
	const CudaProperties cuda_props;

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
	 * Returns the number of threadblocks that the specified device should 
	 * launch for [up|down]sweep grids for the given problem size
	 */
	template <typename MemcopyConfig>
	int SweepGridSize(int num_elements, int max_grid_size) 
	{
		const int SCHEDULE_GRANULARITY = 1 << MemcopyConfig::LOG_SCHEDULE_GRANULARITY;

		int default_sweep_grid_size;
		if (cuda_props.device_sm_version < 120) {
			
			// G80/G90: Four times the SM-count
			default_sweep_grid_size = cuda_props.device_props.multiProcessorCount * 4;
			
		} else if (cuda_props.device_sm_version < 200) {
			
			// GT200: Special sauce
			
			// Start with with full downsweep occupancy of all SMs 
			default_sweep_grid_size = 
				cuda_props.device_props.multiProcessorCount * MemcopyConfig::CTA_OCCUPANCY;

			// Increase by default every 64 million key-values
			int step = 1024 * 1024 * 64;		 
			default_sweep_grid_size *= (num_elements + step - 1) / step;

			double multiplier1 = 4.0;
			double multiplier2 = 16.0;

			double delta1 = 0.068;
			double delta2 = 0.1285;
			
			int dividend = (num_elements + 512 - 1) / 512;

			int bumps = 0;
			while(true) {

				if (default_sweep_grid_size <= cuda_props.device_props.multiProcessorCount) {
					break;
				}
				
				double quotient = ((double) dividend) / (multiplier1 * default_sweep_grid_size);
				quotient -= (int) quotient;

				if ((quotient > delta1) && (quotient < 1 - delta1)) {

					quotient = ((double) dividend) / (multiplier2 * default_sweep_grid_size / 3.0);
					quotient -= (int) quotient;

					if ((quotient > delta2) && (quotient < 1 - delta2)) {
						break;
					}
				}

				if (bumps == 3) {
					// Bump it down by 27
					default_sweep_grid_size -= 27;
					bumps = 0;
				} else {
					// Bump it down by 1
					default_sweep_grid_size--;
					bumps++;
				}
			}
			
		} else {

			// GF10x
			if (cuda_props.device_sm_version == 210) {
				// GF110
				default_sweep_grid_size = 4 * (cuda_props.device_props.multiProcessorCount * MemcopyConfig::CTA_OCCUPANCY);
			} else {
				// Anything but GF110
				default_sweep_grid_size = 4 * (cuda_props.device_props.multiProcessorCount * MemcopyConfig::CTA_OCCUPANCY) - 2;
			}
		}
		
		// Reduce by override, if specified
		if (max_grid_size > 0) {
			default_sweep_grid_size = max_grid_size;
		}
		
		// Reduce if we have less work than we can divide up among this 
		// many CTAs
		
		int grains = (num_elements + SCHEDULE_GRANULARITY - 1) / SCHEDULE_GRANULARITY;
		if (default_sweep_grid_size > grains) {
			default_sweep_grid_size = grains;
		}
		
		return default_sweep_grid_size;
	}	


public:

	/**
	 * Debug level.  If set, the enactor blocks after kernel calls to check
	 * for successful launch/execution
	 */
	bool DEBUG;


	/**
	 * Constructor.
	 */
	MemcopyEnactor() :
			d_work_progress(NULL),
			progress_selector(0),
#if	defined(__THRUST_SYNCHRONOUS) || defined(DEBUG) || defined(_DEBUG)
			DEBUG(true)
#else
			DEBUG(false)
#endif
		{}


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
		typedef MemcopyKernelConfig<MemcopyConfig> MemcopyKernelConfigType;

		const int SCHEDULE_GRANULARITY 	= 1 << MemcopyConfig::LOG_SCHEDULE_GRANULARITY;

		int sweep_grid_size = SweepGridSize<MemcopyConfig>(length, max_grid_size);
		int dynamic_smem = 0;
		
		CtaWorkDistribution<SizeT> work(length, SCHEDULE_GRANULARITY, sweep_grid_size);

		if (DEBUG) {
			printf("\n\n");
			printf("CodeGen: \t[device_sm_version: %d, kernel_ptx_version: %d]\n", 
				cuda_props.device_sm_version, cuda_props.kernel_ptx_version);
			printf("Memcopy: \t[grid_size: %d, threads %d]\n",
				sweep_grid_size, 1 << MemcopyConfig::LOG_THREADS);
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
			MemcopyKernel<MemcopyKernelConfigType><<<sweep_grid_size, MemcopyKernelConfigType::THREADS, dynamic_smem>>>(
				d_dest, d_src, d_work_progress, work, progress_selector);
			if (DEBUG && (retval = B40CPerror(cudaThreadSynchronize(),
				"MemcopyEnactor:: MemcopyKernel failed ", __FILE__, __LINE__))) break;

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

