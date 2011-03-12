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
 * WarpRakeAndScan
 ******************************************************************************/

#pragma once

#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/warp_scan.cuh>
#include <b40c/util/scan/serial_scan.cuh>

namespace b40c {
namespace util {
namespace scan {

/**
 * Warp rake and scan. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 */
template <
	typename T,
	T ScanOp(const T&, const T&),
	int PARTIALS_PER_SEG,
	int LOG_RAKING_THREADS,
	bool EXCLUSIVE>
__device__ __forceinline__ void WarpRakeAndScan(
	T *raking_seg,
	volatile T warpscan[][1 << LOG_RAKING_THREADS])
{
	const int RAKING_THREADS = 1 << LOG_RAKING_THREADS;

	if (threadIdx.x < RAKING_THREADS) {

		// Raking reduction
		T inclusive_partial = reduction::SerialReduce<T, PARTIALS_PER_SEG, ScanOp>::Invoke(raking_seg);

		// Exclusive warp scan
		T exclusive_partial = WarpScan<T, LOG_RAKING_THREADS, true, LOG_RAKING_THREADS, ScanOp>::Invoke(
			inclusive_partial, warpscan);

		// Raking scan
		SerialScan<T, PARTIALS_PER_SEG, EXCLUSIVE, ScanOp>::Invoke(raking_seg, exclusive_partial);
	}
}


/**
 * Warp rake and scan. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 *
 * Carry is updated in all raking threads
 */
template <
	typename T,
	T ScanOp(const T&, const T&),
	int PARTIALS_PER_SEG,
	int LOG_RAKING_THREADS,
	bool EXCLUSIVE>
__device__ __forceinline__ void WarpRakeAndScan(
	T *raking_seg,
	volatile T warpscan[][1 << LOG_RAKING_THREADS],
	T &carry)
{
	const int RAKING_THREADS = 1 << LOG_RAKING_THREADS;

	if (threadIdx.x < RAKING_THREADS) {

		// Raking reduction
		T inclusive_partial = reduction::SerialReduce<T, PARTIALS_PER_SEG, ScanOp>::Invoke(raking_seg);

		// Exclusive warp scan
		T warpscan_total;
		T exclusive_partial = WarpScan<T, LOG_RAKING_THREADS, true, LOG_RAKING_THREADS, ScanOp>::Invoke(
				inclusive_partial, warpscan_total, warpscan);
		exclusive_partial = ScanOp(exclusive_partial, carry);

		// Raking scan
		SerialScan<T, PARTIALS_PER_SEG, EXCLUSIVE, ScanOp>::Invoke(raking_seg, exclusive_partial);

		carry = ScanOp(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
	}
}


} // namespace scan
} // namespace util
} // namespace b40c

