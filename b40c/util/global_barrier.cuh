/******************************************************************************
 * 
 * Copyright 2010 Duane Merrill
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
 * Software Global Barrier
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>

namespace b40c {
namespace util {


/**
 * Simple wrapper for returning a CG-loaded int at the specified pointer  
 */
__device__ __forceinline__ int LoadCG(int* d_ptr) 
{
	int retval;
	ModifiedLoad<util::io::ld::cg>::Ld(retval, d_ptr);
	return retval;
}


/**
 * Implements a global, lock-free software barrier between gridDim.x CTAs using 
 * the synchronization vector d_sync having size gridDim.x 
 */
__device__ __forceinline__ void GlobalBarrier(int* d_sync) 
{
	// Threadfence and syncthreads to make sure global writes are visible before 
	// thread-0 reports in with its sync counter
	__threadfence();
	__syncthreads();
	
	if (blockIdx.x == 0) {

		// Report in ourselves
		if (threadIdx.x == 0) {
			d_sync[blockIdx.x] = 1; 
		}

		__syncthreads();
		
		// Wait for everyone else to report in
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
			while (LoadCG(d_sync + peer_block) == 0) {
				__threadfence_block();
			}
		}

		__syncthreads();
		
		// Let everyone know it's safe to read their prefix sums
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
			d_sync[peer_block] = 0;
		}

	} else {
		
		if (threadIdx.x == 0) {
			// Report in 
			d_sync[blockIdx.x] = 1; 

			// Wait for acknowledgement
			while (LoadCG(d_sync + blockIdx.x) == 1) {
				__threadfence_block();
			}
		}
		
		__syncthreads();
	}
}


} // namespace util
} // namespace b40c

