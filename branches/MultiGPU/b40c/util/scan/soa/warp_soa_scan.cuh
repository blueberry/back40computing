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
 * WarpSoaScan
 ******************************************************************************/

#pragma once

#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace util {
namespace scan {
namespace soa {

/**
 * Structure-of-arrays tuple warpscan.  Performs NUM_ELEMENTS steps of a
 * Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 */
template <
	typename Tuple,						// Tuple of partials
	typename WarpscanSoa,				// Tuple of SOA warpscan segments
	int LOG_NUM_ELEMENTS,
	bool EXCLUSIVE,
	int STEPS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct WarpSoaScan;


/**
 * Inclusive WarpSoaScan
 */
template <
	typename Tuple,						// Tuple of partials
	typename WarpscanSoa,				// Tuple of SOA warpscan segments
	int LOG_NUM_ELEMENTS,
	int STEPS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct WarpSoaScan<Tuple, WarpscanSoa, LOG_NUM_ELEMENTS, false, STEPS, ScanOp>
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_LEFT, int WIDTH>
	struct Iterate
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
			int warpscan_tid)
		{
			// Store exclusive partial
			warpscan_partials.template Set<1>(exclusive_partial, warpscan_tid);

			// Load current partial
			Tuple current_partial = warpscan_partials.template Get<1, Tuple>(
				warpscan_tid - OFFSET_LEFT);

			// Compute inclusive partial from exclusive and current partials
			Tuple inclusive_partial = ScanOp(current_partial, exclusive_partial);

			// Recurse
			return Iterate<OFFSET_LEFT * 2, WIDTH>::Invoke(
				inclusive_partial, warpscan_partials, warpscan_tid);
		}
	};

	// Termination
	template <int WIDTH>
	struct Iterate<WIDTH, WIDTH>
	{
		static __host__ __device__ __forceinline__ Tuple Invoke(
			Tuple exclusive_partial,
			WarpscanSoa warpscan_partials,
			int warpscan_tid)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		Tuple current_partial,						// Input partial
		WarpscanSoa warpscan_partials,				// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;
		return Iterate<1, WIDTH>::Invoke(current_partial, warpscan_partials, warpscan_tid);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		Tuple current_partial,						// Input partial
		Tuple &total_reduction,						// Total reduction (out param)
		WarpscanSoa warpscan_partials,				// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		Tuple inclusive_partial = Invoke(current_partial, warpscan_partials, warpscan_tid);

		// Write our inclusive partial
		warpscan_partials.template Set<1>(inclusive_partial, warpscan_tid);

		// Set total to the last thread's inclusive partial
		total_reduction = warpscan_partials.template Get<1, Tuple>(NUM_ELEMENTS - 1);

		return inclusive_partial;
	}
};


/**
 * Exclusive WarpSoaScan
 */
template <
	typename Tuple,						// Tuple of partials
	typename WarpscanSoa,				// Tuple of SOA warpscan segments
	int LOG_NUM_ELEMENTS,
	int STEPS,
	Tuple ScanOp(Tuple&, Tuple&)>
struct WarpSoaScan<Tuple, WarpscanSoa, LOG_NUM_ELEMENTS, true, STEPS, ScanOp>
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		Tuple current_partial,						// Input partial
		WarpscanSoa warpscan_partials,				// SOA of tuples in smem for warpscanning.  Each array contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		Tuple inclusive_partial = WarpSoaScan<
			Tuple,
			WarpscanSoa,
			LOG_NUM_ELEMENTS,
			false,
			STEPS,
			ScanOp>::Invoke(current_partial, warpscan_partials, warpscan_tid);

		// Write our inclusive partial
		warpscan_partials.template Set<1>(inclusive_partial, warpscan_tid);

		// Return exclusive partial
		return warpscan_partials.template Get<1, Tuple>(warpscan_tid - 1);
	}

	// Interface
	static __host__ __device__ __forceinline__ Tuple Invoke(
		Tuple current_partial,						// Input partial
		Tuple &total_reduction,						// Total reduction (out param)
		WarpscanSoa warpscan_partials,				// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		Tuple inclusive_partial = WarpSoaScan<
			Tuple,
			WarpscanSoa,
			LOG_NUM_ELEMENTS,
			false,
			STEPS,
			ScanOp>::Invoke(current_partial, warpscan_partials, warpscan_tid);

		// Write our inclusive partial
		warpscan_partials.template Set<1>(inclusive_partial, warpscan_tid);

		// Set total to the last thread's inclusive partial
		total_reduction = warpscan_partials.template Get<1, Tuple>(NUM_ELEMENTS - 1);

		// Return exclusive partial
		return warpscan_partials.template Get<1, Tuple>(warpscan_tid - 1);
	}
};



} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

