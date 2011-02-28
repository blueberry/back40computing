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
 * Perform a warp-synchronous reduction.
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
	T BinaryOp(const T&, const T&) = defaults::Sum<T> >
struct WarpReduce
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_RIGHT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(T partial, volatile T *storage, int tid) 
		{
			T from_storage = storage[tid + OFFSET_RIGHT];
			partial = BinaryOp(partial, from_storage);
			storage[tid] = partial;
			Iterate<OFFSET_RIGHT / 2>::Invoke(partial, storage, tid);
		}
	};
	
	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		static __device__ __forceinline__ void Invoke(T partial, volatile T *storage, int tid) {}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,					// Input partial
		volatile T *storage,		// Smem for reducing of length equal to at least 1.5x NUM_ELEMENTS
		int tid = threadIdx.x)		// Thread's local index into a segment of NUM_ELEMENTS items
	{
		storage[tid] = partial;
		Iterate<NUM_ELEMENTS / 2>::Invoke(partial, storage, tid);
		return storage[0];
	}
};



/**
 * Have each thread concurrently perform a serial reduction over its specified segment 
 */
template <
	typename T,
	int LENGTH,
	T BinaryOp(const T&, const T&) = defaults::Sum<T> >
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
 * Collective reduction across all threads: One-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have one warp or smaller of raking threads.
 */
template <
	typename T,
	typename SrtsGrid,
	T BinaryOp(const T&, const T&),
	util::st::CacheModifier WRITE_MODIFIER>
__device__ __forceinline__ void CollectiveReduction(
	T partial,
	T *out,
	T *grid)
{
	// Determine the deposit and raking pointers for SRTS grid
	T *primary_base_partial = SrtsGrid::BasePartial(grid);

	// Place partials in primary smem grid
	primary_base_partial[0] = partial;

	__syncthreads();

	// Primary rake and reduce (guaranteed one warp or fewer raking threads)
	if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

		// Raking reduction
		T *primary_raking_seg = SrtsGrid::RakingSegment(grid);
		T raking_partial = SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(primary_raking_seg);

		// WarpReduce
		T total = WarpReduce<T, SrtsGrid::LOG_RAKING_THREADS, BinaryOp>::Invoke(
			raking_partial, grid);

		// Write output
		if (threadIdx.x == 0) {
			util::ModifiedStore<T, WRITE_MODIFIER>::St(total, out, 0);
		}
	}
}


/**
 * Collective reduction across all threads: Two-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have more than one warp of raking threads.
 */
template <
	typename T,
	typename PrimarySrtsGrid,
	typename SecondarySrtsGrid,
	T BinaryOp(const T&, const T&),
	util::st::CacheModifier WRITE_MODIFIER>
__device__ __forceinline__ void CollectiveReduction(
	T partial,
	T *out,
	T *primary_grid,
	T *secondary_grid)
{
	// Determine the deposit and raking pointers for SRTS grids
	T *primary_base_partial = PrimarySrtsGrid::BasePartial(primary_grid);

	// Place partials in primary smem grid
	primary_base_partial[0] = partial;

	__syncthreads();

	// Primary rake and reduce
	if (threadIdx.x < PrimarySrtsGrid::RAKING_THREADS) {

		// Raking reduction in primary grid
		T *primary_raking_seg = PrimarySrtsGrid::RakingSegment(primary_grid);
		T raking_partial = SerialReduce<T, PrimarySrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(primary_raking_seg);

		// Place raked partial in secondary grid
		T *secondary_base_partial = SecondarySrtsGrid::BasePartial(secondary_grid);
		secondary_base_partial[0] = raking_partial;
	}

	__syncthreads();

	// Secondary rake and reduce (guaranteed one warp or fewer raking threads)
	if (threadIdx.x < SecondarySrtsGrid::RAKING_THREADS) {

		// Raking reduction in secondary grid
		T *secondary_raking_seg = SecondarySrtsGrid::RakingSegment(secondary_grid);
		T raking_partial = SerialReduce<T, SecondarySrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(secondary_raking_seg);

		// WarpReduce
		T total = WarpReduce<T, SecondarySrtsGrid::LOG_RAKING_THREADS, BinaryOp>::Invoke(
			raking_partial, secondary_grid);

		// Write output
		if (threadIdx.x == 0) {
			util::ModifiedStore<T, WRITE_MODIFIER>::St(total, out, 0);
		}
	}
};


} // namespace reduction
} // namespace b40c

