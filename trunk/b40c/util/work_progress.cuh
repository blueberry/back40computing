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
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Management of temporary device storage needed for implementing work-stealing
 * progress between CTAs in a single grid
 ******************************************************************************/

#pragma once

#include <b40c/util/error_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/device_intrinsics.cuh>

namespace b40c {
namespace util {

/**
 * Manages device storage needed for implementing work-stealing
 * progress between CTAs in a single grid
 *
 * Consists of a pair of counters in global device memory and a selector for
 * indexing into the pair.  If we perform work-stealing passes, the
 * current counter provides an atomic reference of progress.  One pass
 * resets the counter for the next.
 */
class WorkProgress
{
protected :

	// Pair of counters in global device memory
	size_t*		d_work_progress;

	// Selector for indexing into the pair
	int 		progress_selector;

public:

	/**
	 * Constructor
	 */
	WorkProgress() : d_work_progress(NULL), progress_selector(0) {}

	/**
	 * Steals TILE_ELEMENTS of work from the current progress counter, returning the
	 * offset of that work (from zero).  Typically called by thread-0
	 */
	template <int TILE_ELEMENTS>
	__device__ __forceinline__ size_t Steal() const
	{
		return util::AtomicInt<size_t>::Add(d_work_progress + progress_selector, TILE_ELEMENTS);
	}


	/**
	 * Resets the work progress for the next pass.  Typically called by thread-0 in block-0.
	 */
	__device__ __forceinline__ void PrepareNext() const
	{
		d_work_progress[progress_selector ^ 1] = 0;
	}

};


/**
 * Version of work progress with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base WorkProgress
 * as parameters to kernels.
 */
class WorkProgressLifetime : public WorkProgress
{
public:

	/**
	 * Constructor
	 */
	WorkProgressLifetime() : WorkProgress() {}


	/**
	 * Deallocates and resets the progress counters
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		if (d_work_progress) {
			retval = util::B40CPerror(cudaFree(d_work_progress), "WorkProgress cudaFree d_work_progress failed: ", __FILE__, __LINE__);
			d_work_progress = NULL;
		}
		progress_selector = 0;
		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~WorkProgressLifetime()
	{
		HostReset();
	}


	/**
	 * Sets up the progress counters for the next kernel launch (lazily
	 * allocating and initializing them if necessary)
	 */
	cudaError_t Setup()
	{
		cudaError_t retval = cudaSuccess;
		do {
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
		} while (0);

		return retval;
	}
};

} // namespace util
} // namespace b40c

