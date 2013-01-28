/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Software global barrier for CTAs
 ******************************************************************************/

#pragma once

#include "../debug.cuh"
#include "../ns_wrapper.cuh"
#include "../thread/thread_load.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Manages device storage needed for implementing a global software barrier
 * between CTAs in a single grid
 */
class GridTestBarrier
{
protected :

	typedef unsigned int SyncFlag;

	// Counters in global device memory
	SyncFlag* d_sync;

public:

	/**
	 * Constructor
	 */
	GridTestBarrier() : d_sync(NULL) {}


	/**
	 * Synchronize
	 */
	__device__ __forceinline__ void Sync() const
	{
	    volatile SyncFlag *d_vol_sync = d_sync;

		// Threadfence and syncthreads to make sure global writes are visible before
		// thread-0 reports in with its sync counter
		__threadfence();
		__syncthreads();

		if (blockIdx.x == 0)
		{
			// Report in ourselves
			if (threadIdx.x == 0)
			{
			    d_vol_sync[blockIdx.x] = 1;
			}

			__syncthreads();

			// Wait for everyone else to report in
			for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x)
			{
				while (ThreadLoad<PTX_LOAD_CG>(d_sync + peer_block) == 0)
				{
					__threadfence_block();
				}
			}

			__syncthreads();

			// Let everyone know it's safe to read their prefix sums
			for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x)
			{
			    d_vol_sync[peer_block] = 0;
			}
		}
		else
		{
			if (threadIdx.x == 0)
			{
				// Report in
			    d_vol_sync[blockIdx.x] = 1;

				// Wait for acknowledgement
				while (ThreadLoad<PTX_LOAD_CG>(d_sync + blockIdx.x) == 1)
				{
					__threadfence_block();
				}
			}

			__syncthreads();
		}
	}
};


/**
 * Version of global barrier with storage lifetime management.
 *
 * Uses RAII for lifetime, i.e., device resources are reclaimed when
 * the destructor is called (e.g., when the logical scope ends).
 */
class GridTestBarrierLifetime : public GridTestBarrier
{
protected:

	// Number of bytes backed by d_sync
	size_t sync_bytes;

public:

	/**
	 * Constructor
	 */
	GridTestBarrierLifetime() : GridTestBarrier(), sync_bytes(0) {}


	/**
	 * DeviceFrees and resets the progress counters
	 */
	cudaError_t HostReset()
	{
		cudaError_t retval = cudaSuccess;
		if (d_sync)
		{
			retval = CubDebug(cudaFree(d_sync));
			d_sync = NULL;
		}
		sync_bytes = 0;
		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~GridTestBarrierLifetime()
	{
		HostReset();
	}


	/**
	 * Sets up the progress counters for the next kernel launch (lazily
	 * allocating and initializing them if necessary)
	 */
	cudaError_t Setup(int sweep_grid_size)
	{
		cudaError_t retval = cudaSuccess;
		do {
			size_t new_sync_bytes = sweep_grid_size * sizeof(SyncFlag);
			if (new_sync_bytes > sync_bytes)
			{
				if (d_sync)
				{
					if (retval = CubDebug(cudaFree(d_sync))) break;
				}

				sync_bytes = new_sync_bytes;

				// Allocate and initialize to zero
				if (retval = CubDebug(cudaMalloc((void**) &d_sync, sync_bytes))) break;
				if (retval = CubDebug(cudaMemset(d_sync, 0, new_sync_bytes))) break;
			}
		} while (0);

		return retval;
	}
};


} // namespace cub
CUB_NS_POSTFIX
