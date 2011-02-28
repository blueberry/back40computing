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

namespace b40c {
namespace scan {


namespace defaults {

/**
 * Binary associative operator for addition
 */
template <typename T>
T __host__ __device__ __forceinline__ Sum(const T &a, const T &b)
{
	return a + b;
}

} // defaults




/**
 * Performs NUM_ELEMENTS steps of a Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	int STEPS = LOG_NUM_ELEMENTS,
	T BinaryOp(const T&, const T&) = defaults::Sum<T> >
struct WarpScanInclusive
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;
	static const int WIDTH = 1 << STEPS;

	// General iteration
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ int Invoke(
			T partial, volatile T ks_warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			ks_warpscan[1][warpscan_tid] = partial;
			partial = BinaryOp(partial, ks_warpscan[1][warpscan_tid - OFFSET_LEFT]);
			return Iterate<OFFSET_LEFT * 2>::Invoke(partial, ks_warpscan, warpscan_tid);
		}
	};

	// Termination
	template <int __dummy>
	struct Iterate<WIDTH, __dummy>
	{
		static __device__ __forceinline__ int Invoke(
			T partial, volatile T ks_warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			return partial;
		}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		volatile T ks_warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Iterate<1>::Invoke(partial, ks_warpscan, warpscan_tid);
	}
};


/**
 * Performs NUM_ELEMENTS steps of a Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * Can be used to perform concurrent, independent warp-scans if
 * storage pointers and their local-thread indexing id's are set up properly.
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	T BinaryOp(const T&, const T&) = defaults::Sum<T> >
struct WarpScan
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

			partial = BinaryOp(partial, warpscan[1][warpscan_tid - OFFSET_LEFT]);
			warpscan[1][warpscan_tid] = partial;
			Iterate<OFFSET_LEFT * 2>::Invoke(partial, warpscan, warpscan_tid);
		}
	};

	// Termination
	template <int __dummy>
	struct Iterate<NUM_ELEMENTS, __dummy>
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS],
			int warpscan_tid) {}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		T &total,									// Total aggregate reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		warpscan[1][warpscan_tid] = partial;

		Iterate<1>::Invoke(partial, warpscan, warpscan_tid);

		// Set aggregate reduction
		total = warpscan[1][NUM_ELEMENTS - 1];

		// Return scan partial
		return warpscan[1][warpscan_tid - 1];
	}
};


/**
 * Have each thread concurrently perform a serial scan over its
 * specified segment (in place).  Returns the inclusive total.
 */
template <
	typename T,
	int LENGTH,
	T BinaryOp(const T&, const T&) = defaults::Sum<T> >
struct SerialScan
{
	// Iterate
	template <int COUNT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial)
		{
			T inclusive_partial = BinaryOp(partials[COUNT], exclusive_partial);
			results[COUNT] = exclusive_partial;
			return Iterate<COUNT + 1>::Invoke(partials, results, inclusive_partial);
		}
	};

	// Terminate
	template <int __dummy>
	struct Iterate<LENGTH, __dummy>
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
 * Warp rake and scan. Must hold that the number of raking threads in the grid
 * config type is at most the size of a warp.  (May be less.)
 */
template <typename SrtsGrid>
__device__ __forceinline__ void WarpRakeAndScan(
	typename SrtsGrid::PartialType *raking_seg,
	typename SrtsGrid::PartialType warpscan[2][SrtsGrid::RAKING_THREADS],
	typename SrtsGrid::PartialType &carry)
{
	typedef typename SrtsGrid::PartialType PartialType;

	if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

		// Raking reduction
		PartialType partial = SerialReduce<PartialType, SrtsGrid::PARTIALS_PER_SEG>::Invoke(raking_seg);

		// Warpscan
		PartialType warpscan_total;
		partial = WarpScan<PartialType, SrtsGrid::LOG_RAKING_THREADS>::Invoke(partial, warpscan_total, warpscan);
		partial += carry;

		// Raking scan
		SerialScan<PartialType, SrtsGrid::PARTIALS_PER_SEG>::Invoke(raking_seg, partial);

		carry += warpscan_total;			// Increment the CTA's running total by the full tile reduction
	}
}


} // namespace scan
} // namespace b40c

