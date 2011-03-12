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
 * WarpReduce
 ******************************************************************************/

#pragma once

#include <b40c/util/reduction/operators.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Perform NUM_ELEMENTS of warp-synchronous reduction.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * Can be used to perform concurrent, independent warp-reductions if
 * storage pointers and their local-thread indexing id's are set up properly.
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	T ReductionOp(const T&, const T&) = DefaultSum>
struct WarpReduce
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS],
			int warpscan_tid)
		{
			T offset_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];
			partial = ReductionOp(partial, offset_partial);
			warpscan[1][warpscan_tid] = partial;
			Iterate<OFFSET_LEFT / 2>::Invoke(partial, warpscan, warpscan_tid);
		}
	};
	
	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS],
			int warpscan_tid) {}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,								// Input partial
		volatile T warpscan[][NUM_ELEMENTS],	// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		warpscan[1][warpscan_tid] = partial;
		Iterate<NUM_ELEMENTS / 2>::Invoke(partial, warpscan, warpscan_tid);

		// Return aggregate reduction
		return warpscan[1][NUM_ELEMENTS - 1];
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c

