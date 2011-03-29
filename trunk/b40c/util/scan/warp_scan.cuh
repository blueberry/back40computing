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
 * WarpScan
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace scan {



/**
 * Performs NUM_ELEMENTS steps of a Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	bool EXCLUSIVE = true,
	int STEPS = LOG_NUM_ELEMENTS,
	T ScanOp(const T&, const T&) = DefaultSum>
struct WarpScan;


/**
 * Inclusive warpscan
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	int STEPS,
	T ScanOp(const T&, const T&)>
struct WarpScan<T, LOG_NUM_ELEMENTS, false, STEPS, ScanOp>
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// General iteration
	template <int OFFSET_LEFT, int WIDTH>
	struct Iterate
	{
		static __device__ __forceinline__ T Invoke(
			T exclusive_partial,
			volatile T warpscan[][NUM_ELEMENTS],
			int warpscan_tid)
		{
			warpscan[1][warpscan_tid] = exclusive_partial;
			T offset_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];
			T inclusive_partial = ScanOp(offset_partial, exclusive_partial);

			return Iterate<OFFSET_LEFT * 2, WIDTH>::Invoke(inclusive_partial, warpscan, warpscan_tid);
		}
	};

	// Termination
	template <int WIDTH>
	struct Iterate<WIDTH, WIDTH>
	{
		static __device__ __forceinline__ T Invoke(
			T exclusive_partial, volatile T warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			return exclusive_partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Returns inclusive partial
	 */
	static __device__ __forceinline__ T Invoke(
		T current_partial,							// Input partial
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;
		return Iterate<1, WIDTH>::Invoke(current_partial, warpscan, warpscan_tid);
	}

	/**
	 * Returns inclusive partial and cumulative reduction
	 */
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		T &total_reduction,							// Total reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		T inclusive_partial = Invoke(current_partial, warpscan, warpscan_tid);

		// Write our inclusive partial and then set total to the last thread's inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;
		total_reduction = warpscan[1][NUM_ELEMENTS - 1];

		// Return scan partial
		return inclusive_partial;
	}

};


/**
 * Exclusive warpscan
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	int STEPS,
	T ScanOp(const T&, const T&)>
struct WarpScan<T, LOG_NUM_ELEMENTS, true, STEPS, ScanOp>
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Returns exclusive partial
	 */
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		T inclusive_partial = WarpScan<T, LOG_NUM_ELEMENTS, false, STEPS, ScanOp>::Invoke(
			current_partial, warpscan, warpscan_tid);

		// Write out our inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;

		// Return exclusive partial
		return warpscan[1][warpscan_tid - 1];
	}

	/**
	 * Returns exclusive partial and cumulative reduction
	 */
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		T &total_reduction,							// Total reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		T inclusive_partial = WarpScan<T, LOG_NUM_ELEMENTS, false, STEPS, ScanOp>::Invoke(
			current_partial, warpscan, warpscan_tid);

		// Write our inclusive partial and then set total to the last thread's inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;
		total_reduction = warpscan[1][NUM_ELEMENTS - 1];

		// Return exclusive partial
		return warpscan[1][warpscan_tid - 1];
	}
};


} // namespace scan
} // namespace util
} // namespace b40c

