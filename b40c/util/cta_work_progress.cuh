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
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

namespace b40c {
namespace util {

/**
 * Manages device storage needed for implementing work-stealing
 * and queuing progress between CTAs in a single grid.
 *
 * Can be used for:
 *
 * (1) Work-stealing. Consists of a pair of counters in
 *     global device memory, optionally, a host-managed selector for
 *     indexing into the pair.
 *
 * (2) Device-managed queue.  Consists of a quadruplet of counters
 *     in global device memory and selection into the counters is made
 *     based upon the supplied iteration count.
 *     		Current iteration: incoming queue length
 *     		Next iteration: outgoing queue length
 *     		Next next iteration: needs to be reset before next iteration
 *     Can be used with work-stealing counters to for work-stealing
 *     queue operation
 *
 * For work-stealing passes, the current counter provides an atomic
 * reference of progress, and the current pass will typically reset
 * the counter for the next.
 *
 */
class CtaWorkProgress
{
protected :

	static const int QUEUE_COUNTERS = 4;
	static const int STEAL_COUNTERS = 2;

	// Six pointer-sized counters in global device memory (we may not use
	// all of them, or may only use 32-bit versions of them)
	size_t *d_counters;

	// Host-controlled selector for indexing into d_counters.
	int progress_selector;

public:

	static const int COUNTERS = QUEUE_COUNTERS + STEAL_COUNTERS;

	/**
	 * Constructor
	 */
	CtaWorkProgress() : d_counters(NULL), progress_selector(0) {}

	/**
	 * Resets all counters.  Must be called by thread-0 through
	 * thread-(COUNTERS - 1)
	 */
	template <typename SizeT>
	__device__ __forceinline__ void Reset()
	{
		SizeT reset_val = 0;
		util::io::ModifiedStore<util::io::st::cg>::St(
			reset_val, ((SizeT *) d_counters) + threadIdx.x);
	}

	//---------------------------------------------------------------------
	// Work-stealing
	//---------------------------------------------------------------------

	/**
	 * Steals work from the host-indexed progress counter, returning
	 * the offset of that work (from zero) and incrementing it by count.
	 * Typically called by thread-0
	 */
	template <typename SizeT>
	__device__ __forceinline__ SizeT Steal(int count)
	{
		SizeT* d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		return util::AtomicInt<SizeT>::Add(d_steal_counters + progress_selector, count);
	}

	/**
	 * Steals work from the specified iteration's progress counter, returning the
	 * offset of that work (from zero) and incrementing it by count.
	 * Typically called by thread-0
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT Steal(int count, IterationT iteration)
	{
		SizeT* d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		return util::AtomicInt<SizeT>::Add(d_steal_counters + (iteration & 1), count);
	}

	/**
	 * Resets the work progress for the next host-indexed work-stealing
	 * pass.  Typically called by thread-0 in block-0.
	 */
	template <typename SizeT>
	__device__ __forceinline__ void PrepResetSteal()
	{
		SizeT 	reset_val = 0;
		SizeT* 	d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		util::io::ModifiedStore<util::io::st::cg>::St(
				reset_val, d_steal_counters + (progress_selector ^ 1));
	}

	/**
	 * Resets the work progress for the specified work-stealing iteration.
	 * Typically called by thread-0 in block-0.
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ void PrepResetSteal(IterationT iteration)
	{
		SizeT 	reset_val = 0;
		SizeT* 	d_steal_counters = ((SizeT*) d_counters) + QUEUE_COUNTERS;
		util::io::ModifiedStore<util::io::st::cg>::St(
			reset_val, d_steal_counters + (iteration & 1));
	}


	//---------------------------------------------------------------------
	// Queuing
	//---------------------------------------------------------------------

	/**
	 * Get counter for specified iteration
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT* GetQueueCounter(IterationT iteration)
	{
		return ((SizeT*) d_counters) + (iteration & 3);
	}

	/**
	 * Load work queue length for specified iteration
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ SizeT LoadQueueLength(IterationT iteration)
	{
		SizeT queue_length;
		util::io::ModifiedLoad<util::io::ld::cg>::Ld(
			queue_length, GetQueueCounter<SizeT>(iteration));
		return queue_length;
	}

	/**
	 * Store work queue length for specified iteration
	 */
	template <typename SizeT, typename IterationT>
	__device__ __forceinline__ void StoreQueueLength(SizeT queue_length, IterationT iteration)
	{
		util::io::ModifiedStore<util::io::st::cg>::St(
			queue_length, GetQueueCounter<SizeT>(iteration));
	}
};


/**
 * Version of work progress with storage lifetime management.
 *
 * We can use this in host enactors, and pass the base CtaWorkProgress
 * as parameters to kernels.
 */
class CtaWorkProgressLifetime : public CtaWorkProgress
{
public:

	/**
	 * Constructor
	 */
	CtaWorkProgressLifetime() :
		CtaWorkProgress() {}


	/**
	 * Deallocates and resets the progress counters
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		if (d_counters) {

			retval = util::B40CPerror(cudaFree(d_counters),
				"CtaWorkProgress cudaFree d_counters failed: ", __FILE__, __LINE__);
			d_counters = NULL;
		}
		progress_selector = 0;
		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~CtaWorkProgressLifetime()
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
			if (!d_counters) {

				size_t h_counters[COUNTERS];
				for (int i = 0; i < COUNTERS; i++) {
					h_counters[i] = 0;
				}

				// Allocate and initialize
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_counters, sizeof(h_counters)),
					"ReductionEnactor cudaMalloc d_counters failed", __FILE__, __LINE__)) break;
				if (retval = util::B40CPerror(cudaMemcpy(d_counters, h_counters, sizeof(h_counters), cudaMemcpyHostToDevice),
					"ReductionEnactor cudaMemcpy d_counters failed", __FILE__, __LINE__)) break;
			}

			// Update our progress counter selector to index the next progress counter
			progress_selector ^= 1;

		} while (0);

		return retval;
	}


	/**
	 * Acquire work queue length
	 */
	template <typename SizeT, typename IterationT>
	cudaError_t GetQueueLength(
		IterationT iteration,
		SizeT &queue_length)		// out param
	{
		cudaError_t retval;
		int queue_length_idx = iteration & 0x3;

		if (retval = cudaMemcpy(
			&queue_length,
			((SizeT*) d_counters) + queue_length_idx,
			1 * sizeof(SizeT),
			cudaMemcpyDeviceToHost))
		{
			printf("cudaMemcpy failed: %d %d", __FILE__, __LINE__);
		}

		return retval;
	}
};

} // namespace util
} // namespace b40c

