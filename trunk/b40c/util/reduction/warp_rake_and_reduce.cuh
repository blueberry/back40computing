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
 * WarpRakeAndReduce
 ******************************************************************************/

#pragma once

#include <b40c/util/reduction/warp_reduce.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Warp rake and reduce. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 *
 * Carry is updated in all raking threads
 */
template <
	typename T,
	T ReductionOp(const T&, const T&),
	int PARTIALS_PER_SEG,
	int LOG_RAKING_THREADS>
__device__ __forceinline__ void WarpRakeAndReduce(
	T *raking_seg,
	volatile T warpscan[][1 << LOG_RAKING_THREADS],
	T &carry)
{
	const int RAKING_THREADS = 1 << LOG_RAKING_THREADS;

	if (threadIdx.x < RAKING_THREADS) {

		// Raking reduction
		T partial = SerialReduce<T, PARTIALS_PER_SEG, ReductionOp>::Invoke(raking_seg);

		// Warp reduction
		T warpscan_total = WarpReduce<T, LOG_RAKING_THREADS, ReductionOp>::Invoke(
			partial, warpscan);
		carry = ReductionOp(carry, warpscan_total);
	}
}


/**
 * Warp rake and reduce. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 *
 * Result is returned in all threads.
 */
template <
	typename T,
	T ReductionOp(const T&, const T&),
	int PARTIALS_PER_SEG,
	int LOG_RAKING_THREADS>
__device__ __forceinline__ T WarpRakeAndReduce(
	T *raking_seg,
	volatile T warpscan[][1 << LOG_RAKING_THREADS])
{
	const int RAKING_THREADS = 1 << LOG_RAKING_THREADS;

	if (threadIdx.x < RAKING_THREADS) {

		// Raking reduction
		T partial = SerialReduce<T, PARTIALS_PER_SEG, ReductionOp>::Invoke(raking_seg);

		// Warp reduction
		WarpReduce<T, LOG_RAKING_THREADS, ReductionOp>::Invoke(partial, warpscan);
	}

	__syncthreads();

	// Return last thread's inclusive partial
	return warpscan[1][RAKING_THREADS - 1];
}


} // namespace reduction
} // namespace util
} // namespace b40c


