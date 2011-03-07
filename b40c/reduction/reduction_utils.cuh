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
 * Simple reduction utilities
 ******************************************************************************/

#pragma once

#include <b40c/util/data_movement_store.cuh>

namespace b40c {
namespace reduction {


namespace defaults {

/**
 * Addition binary associative operator
 */
template <typename T>
T __host__ __device__ __forceinline__ Sum(const T &a, const T &b)
{
	return a + b;
}

} // defaults


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
	T BinaryOp(const T&, const T&) = defaults::Sum>
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
			partial = BinaryOp(partial, offset_partial);
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



/**
 * Have each thread concurrently perform a serial reduction over its specified segment 
 */
template <
	typename T,
	int LENGTH,
	T BinaryOp(const T&, const T&) = defaults::Sum >
struct SerialReduce
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate 
	{
		static __device__ __forceinline__ T Invoke(T partials[]) 
		{
			T a = Iterate<COUNT - 2, TOTAL>::Invoke(partials);
			T b = partials[TOTAL - COUNT];
			T c = partials[TOTAL - (COUNT - 1)];

			// TODO: consider specializing with a video 3-op instructions on SM2.0+, e.g., asm("vadd.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(a) : "r"(a), "r"(b), "r"(c));
			return BinaryOp(a, BinaryOp(b, c));
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		static __device__ __forceinline__ T Invoke(T partials[])
		{
			return BinaryOp(partials[TOTAL - 2], partials[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		static __device__ __forceinline__ T Invoke(T partials[]) 
		{
			return partials[TOTAL - 1];
		}
	};
	
	// Interface
	static __device__ __forceinline__ T Invoke(T partials[])			
	{
		return Iterate<LENGTH, LENGTH>::Invoke(partials);
	}
};


/**
 * Warp rake and reduce. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 *
 * Carry is updated in all raking threads
 */
template <
	typename SrtsGrid,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
__device__ __forceinline__ void WarpRakeAndReduce(
	typename SrtsGrid::T *raking_seg,
	volatile typename SrtsGrid::T warpscan[][SrtsGrid::RAKING_THREADS],
	typename SrtsGrid::T &carry)
{
	typedef typename SrtsGrid::T T;

	if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

		// Raking reduction
		T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(raking_seg);

		// Warp reduction
		T warpscan_total = WarpReduce<T, SrtsGrid::LOG_RAKING_THREADS, BinaryOp>::Invoke(
			partial, warpscan);
		carry = BinaryOp(carry, warpscan_total);
	}
}


/**
 * Warp rake and reduce. Must hold that the number of raking threads in the SRTS
 * grid type is at most the size of a warp.  (May be less.)
 *
 * Result is computed in all threads.
 */
template <
	typename SrtsGrid,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
__device__ __forceinline__ typename SrtsGrid::T WarpRakeAndReduce(
	typename SrtsGrid::T *raking_seg,
	volatile typename SrtsGrid::T warpscan[][SrtsGrid::RAKING_THREADS])
{
	typedef typename SrtsGrid::T T;

	if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

		// Raking reduction
		T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(raking_seg);

		// Warp reduction
		WarpReduce<T, SrtsGrid::LOG_RAKING_THREADS, BinaryOp>::Invoke(partial, warpscan);
	}

	__syncthreads();

	return warpscan[1][SrtsGrid::RAKING_THREADS - 1];
}


} // namespace reduction
} // namespace b40c

