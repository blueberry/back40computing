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
 * Simple reduction utilities
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/******************************************************************************
 * WarpSoaReduce
 ******************************************************************************/

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
	typename Tuple,						// Tuple of partials
	typename WarpscanSoa,				// Tuple of SOA warpscan segments
	int LOG_NUM_ELEMENTS,
	Tuple ReductionOp(Tuple&, Tuple&)>
struct WarpSoaReduce
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ Tuple Invoke(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
			int warpscan_tid)
		{
			// Store exclusive partial
			warpscan_partials.template Set<1>(exclusive_partial, warpscan_tid);

			// Load current partial
			Tuple current_partial = warpscan_partials.template Get<1, Tuple>(warpscan_tid - OFFSET_LEFT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ReductionOp(current_partial, exclusive_partial);

			// Recurse
			return Iterate<OFFSET_LEFT / 2>::Invoke(inclusive_partial, warpscan_partials, warpscan_tid);
		}
	};

	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		static __device__ __forceinline__ Tuple Invoke(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
			int warpscan_tid)
		{
			return exclusive_partial;
		}
	};

	// Interface (result is returned in all warpscan threads)
	static __device__ __forceinline__ Tuple Invoke(
		Tuple current_partial,					// Input partial
		WarpscanSoa warpscan_partials,			// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		Tuple inclusive_partial = Iterate<NUM_ELEMENTS / 2>::Invoke(
			current_partial, warpscan_partials, warpscan_tid);

		// Write our inclusive partial
		warpscan_partials.template Set<1>(inclusive_partial, warpscan_tid);

		// Return last thread's inclusive partial
		return warpscan_partials.template Get<1, Tuple>(NUM_ELEMENTS - 1);
	}

	// Interface (result is returned in last warpscan thread)
	static __device__ __forceinline__ Tuple InvokeSingle(
		Tuple exclusive_partial,				// Input partial
		WarpscanSoa warpscan_partials,			// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Iterate<NUM_ELEMENTS / 2>::Invoke(exclusive_partial, warpscan_partials, warpscan_tid);
	}

};



} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

