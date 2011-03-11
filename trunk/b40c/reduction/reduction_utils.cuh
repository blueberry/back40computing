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

namespace b40c {
namespace reduction {


/**
 * Addition binary associative operator
 */
template <typename T>
T __host__ __device__ __forceinline__ DefaultSum(const T &a, const T &b)
{
	return a + b;
}


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



/**
 * Have each thread concurrently perform a serial reduction over its specified segment 
 */
template <
	typename T,
	int NUM_ELEMENTS,
	T ReductionOp(const T&, const T&) = DefaultSum >
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
			return ReductionOp(a, ReductionOp(b, c));
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		static __device__ __forceinline__ T Invoke(T partials[])
		{
			return ReductionOp(partials[TOTAL - 2], partials[TOTAL - 1]);
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
		return Iterate<NUM_ELEMENTS, NUM_ELEMENTS>::Invoke(partials);
	}
};


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

	return warpscan[1][RAKING_THREADS - 1];
}


} // namespace reduction
} // namespace b40c

