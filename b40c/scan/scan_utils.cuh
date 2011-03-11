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
 * Simple scan utilities
 ******************************************************************************/

#pragma once

#include <b40c/reduction/reduction_utils.cuh>

namespace b40c {
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
	T ScanOp(const T&, const T&) = reduction::DefaultSum>
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
			T inclusive_partial = ScanOp(exclusive_partial, offset_partial);

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

	// Interface
	static __device__ __forceinline__ T Invoke(
		T exclusive_partial,						// Input partial
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;
		return Iterate<1, WIDTH>::Invoke(exclusive_partial, warpscan, warpscan_tid);
	}

	// Interface
	static __device__ __forceinline__ T Invoke(
		T exclusive_partial,						// Input partial
		T &total_reduction,							// Total reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		T inclusive_partial = Invoke(exclusive_partial, warpscan, warpscan_tid);

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

	// Interface
	static __device__ __forceinline__ T Invoke(
		T exclusive_partial,						// Input partial
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		T inclusive_partial = WarpScan<T, LOG_NUM_ELEMENTS, false, STEPS, ScanOp>::Invoke(
			exclusive_partial, warpscan, warpscan_tid);

		// Write out our inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;

		// Return exclusive partial
		return warpscan[1][warpscan_tid - 1];
	}

	// Interface
	static __device__ __forceinline__ T Invoke(
		T exclusive_partial,						// Input partial
		T &total_reduction,							// Total reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning.  Contains at least two segments of size NUM_ELEMENTS (the first being initialized to identity)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		// Obtain inclusive partial
		T inclusive_partial = WarpScan<T, LOG_NUM_ELEMENTS, false, STEPS, ScanOp>::Invoke(
			exclusive_partial, warpscan, warpscan_tid);

		// Write our inclusive partial and then set total to the last thread's inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;
		total_reduction = warpscan[1][NUM_ELEMENTS - 1];

		// Return exclusive partial
		return warpscan[1][warpscan_tid - 1];
	}
};



/**
 * Have each thread concurrently perform a serial scan over its
 * specified segment (in place).  Returns the inclusive total_reduction.
 */
template <
	typename T,
	int NUM_ELEMENTS,
	bool EXCLUSIVE = true,
	T ScanOp(const T&, const T&) = reduction::DefaultSum>
struct SerialScan;


/**
 * Inclusive serial scan
 */
template <
	typename T,
	int NUM_ELEMENTS,
	T ScanOp(const T&, const T&)>
struct SerialScan <T, NUM_ELEMENTS, false, ScanOp>
{
	// Iterate
	template <int COUNT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial)
		{
			T inclusive_partial = ScanOp(partials[COUNT], exclusive_partial);
			results[COUNT] = inclusive_partial;
			return Iterate<COUNT + 1>::Invoke(partials, results, inclusive_partial);
		}
	};

	// Terminate
	template <int __dummy>
	struct Iterate<NUM_ELEMENTS, __dummy>
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0>::Invoke(partials, partials, exclusive_partial);
	}

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T results[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0>::Invoke(partials, results, exclusive_partial);
	}
};


/**
 * Exclusive serial scan
 */
template <
	typename T,
	int NUM_ELEMENTS,
	T ScanOp(const T&, const T&)>
struct SerialScan <T, NUM_ELEMENTS, true, ScanOp>
{
	// Iterate
	template <int COUNT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial)
		{
			T inclusive_partial = ScanOp(partials[COUNT], exclusive_partial);
			results[COUNT] = exclusive_partial;
			return Iterate<COUNT + 1>::Invoke(partials, results, inclusive_partial);
		}
	};

	// Terminate
	template <int __dummy>
	struct Iterate<NUM_ELEMENTS, __dummy>
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial)
		{
			return exclusive_partial;
		}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0>::Invoke(partials, partials, exclusive_partial);
	}

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T results[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0>::Invoke(partials, results, exclusive_partial);
	}
};


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
		T partial = reduction::SerialReduce<T, PARTIALS_PER_SEG, ScanOp>::Invoke(raking_seg);

		// Warp scan
		partial = WarpScan<T, LOG_RAKING_THREADS, EXCLUSIVE, LOG_RAKING_THREADS, ScanOp>::Invoke(
			partial, warpscan);

		// Raking scan
		SerialScan<T, PARTIALS_PER_SEG, EXCLUSIVE, ScanOp>::Invoke(raking_seg, partial);
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
		T partial = reduction::SerialReduce<T, PARTIALS_PER_SEG, ScanOp>::Invoke(raking_seg);

		// Warp scan
		T warpscan_total;
		partial = WarpScan<T, LOG_RAKING_THREADS, EXCLUSIVE, LOG_RAKING_THREADS, ScanOp>::Invoke(
			partial, warpscan_total, warpscan);
		partial = ScanOp(partial, carry);

		// Raking scan
		SerialScan<T, PARTIALS_PER_SEG, EXCLUSIVE, ScanOp>::Invoke(raking_seg, partial);

		carry = ScanOp(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
	}
}


} // namespace scan
} // namespace b40c

