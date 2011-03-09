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
	bool EXCLUSIVE = true,
	int STEPS = LOG_NUM_ELEMENTS,
	T BinaryOp(const T&, const T&) = defaults::Sum>
struct WarpScan;


/**
 * Inclusive warpscan
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	int STEPS,
	T BinaryOp(const T&, const T&)>
struct WarpScan<T, LOG_NUM_ELEMENTS, false, STEPS, BinaryOp>
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_LEFT, int WIDTH>
	struct Iterate
	{
		static __device__ __forceinline__ int Invoke(
			T partial, volatile T warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			warpscan[1][warpscan_tid] = partial;
			T offset_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];
			partial = BinaryOp(partial, offset_partial);

			return Iterate<OFFSET_LEFT * 2, WIDTH>::Invoke(partial, warpscan, warpscan_tid);
		}
	};

	// Termination
	template <int WIDTH>
	struct Iterate<WIDTH, WIDTH>
	{
		static __device__ __forceinline__ int Invoke(
			T partial, volatile T warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			return partial;
		}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;
		return Iterate<1, WIDTH>::Invoke(partial, warpscan, warpscan_tid);
	}

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		T &total_reduction,							// Total reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		partial = Invoke(partial, warpscan, warpscan_tid);

		// Everyone set total_reduction from last thread's partial
		warpscan[1][warpscan_tid] = partial;
		total_reduction = warpscan[1][NUM_ELEMENTS - 1];

		// Return scan partial
		return partial;
	}

};


/**
 * Exclusive warpscan
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	int STEPS,
	T BinaryOp(const T&, const T&)>
struct WarpScan<T, LOG_NUM_ELEMENTS, true, STEPS, BinaryOp>
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_LEFT, int WIDTH>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS],
			int warpscan_tid)
		{
			T offset_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];
			partial = BinaryOp(partial, offset_partial);
			warpscan[1][warpscan_tid] = partial;

			Iterate<OFFSET_LEFT * 2, WIDTH>::Invoke(partial, warpscan, warpscan_tid);
		}
	};

	// Termination
	template <int WIDTH>
	struct Iterate<WIDTH, WIDTH>
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS],
			int warpscan_tid) {}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		const int WIDTH = 1 << STEPS;

		warpscan[1][warpscan_tid] = partial;
		Iterate<1, WIDTH>::Invoke(partial, warpscan, warpscan_tid);

		// Return scan partial
		return warpscan[1][warpscan_tid - 1];
	}


	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		T &total_reduction,							// Total reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		partial = Invoke(partial, warpscan, warpscan_tid);

		// Everyone set total_reduction from last thread's partial
		total_reduction = warpscan[1][NUM_ELEMENTS - 1];

		// Return scan partial
		return partial;
	}
};


/**
 * Have each thread concurrently perform a serial scan over its
 * specified segment (in place).  Returns the inclusive total_reduction.
 */
template <
	typename T,
	int LENGTH,
	bool EXCLUSIVE = true,
	T BinaryOp(const T&, const T&) = defaults::Sum>
struct SerialScan;


/**
 * Inclusive serial scan
 */
template <
	typename T,
	int LENGTH,
	T BinaryOp(const T&, const T&)>
struct SerialScan <T, LENGTH, false, BinaryOp>
{
	// Iterate
	template <int COUNT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial)
		{
			T inclusive_partial = BinaryOp(partials[COUNT], exclusive_partial);
			results[COUNT] = inclusive_partial;
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
 * Exclusive serial scan
 */
template <
	typename T,
	int LENGTH,
	T BinaryOp(const T&, const T&)>
struct SerialScan <T, LENGTH, true, BinaryOp>
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
 * Warp rake and scan. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 */
template <
	typename SrtsGrid,
	bool EXCLUSIVE,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
__device__ __forceinline__ void WarpRakeAndScan(
	typename SrtsGrid::T *raking_seg,
	volatile typename SrtsGrid::T warpscan[][SrtsGrid::RAKING_THREADS])
{
	typedef typename SrtsGrid::T T;

	if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

		// Raking reduction
		T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(raking_seg);

		// Warp scan
		partial = WarpScan<T, SrtsGrid::LOG_RAKING_THREADS, EXCLUSIVE, SrtsGrid::LOG_RAKING_THREADS, BinaryOp>::Invoke(
			partial, warpscan);

		// Raking scan
		SerialScan<T, SrtsGrid::PARTIALS_PER_SEG, EXCLUSIVE, BinaryOp>::Invoke(raking_seg, partial);
	}
}


/**
 * Warp rake and scan. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 *
 * Carry is updated in all raking threads
 */
template <
	typename SrtsGrid,
	bool EXCLUSIVE,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
__device__ __forceinline__ void WarpRakeAndScan(
	typename SrtsGrid::T *raking_seg,
	volatile typename SrtsGrid::T warpscan[][SrtsGrid::RAKING_THREADS],
	typename SrtsGrid::T &carry)
{
	typedef typename SrtsGrid::T T;

	if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

		// Raking reduction
		T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(raking_seg);

		// Warp scan
		T warpscan_total;
		partial = WarpScan<T, SrtsGrid::LOG_RAKING_THREADS, EXCLUSIVE, SrtsGrid::LOG_RAKING_THREADS, BinaryOp>::Invoke(
			partial, warpscan_total, warpscan);
		partial = BinaryOp(partial, carry);

		// Raking scan
		SerialScan<T, SrtsGrid::PARTIALS_PER_SEG, EXCLUSIVE, BinaryOp>::Invoke(raking_seg, partial);

		carry = BinaryOp(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
	}
}


} // namespace scan
} // namespace b40c

