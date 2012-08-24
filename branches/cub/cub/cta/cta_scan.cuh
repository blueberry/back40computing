/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Cooperative scan abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../warp/warp_scan.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Cooperative prefix scan abstraction for CTAs.
 *
 * Features:
 * 		- Very efficient (only two synchronization barriers).
 * 		- Zero bank conflicts for most types.
 * 		- Supports non-commutative scan operators.
 * 		- Supports scan over strip-mined CTA tiles.  (For a given tile of
 * 			input, each thread acquires SUB_TILES arrays of ITEMS consecutive
 * 			inputs, where the logical stride between a given thread's strips is
 * 			(ITEMS * CTA_THREADS) elements.)
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- The scan type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	typename 	T,						// The scan type
	int 		CTA_THREADS,			// The CTA size in threads
	int 		SUB_TILES = 1>			// The number of consecutive subtiles strip-mined for a larger CTA-wide tile
struct CtaScan
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	/**
	 * Layout type for padded CTA raking grid
	 */
	typedef CtaRakingGrid<CTA_THREADS, T, SUB_TILES> CtaRakingGrid;


	enum
	{
		// The total number of elements that need to be cooperatively reduced
		SHARED_ELEMENTS = CTA_THREADS * SUB_TILES,

		// Number of active warps
		WARPS = (CTA_THREADS + DeviceProps::WARP_THREADS - 1) / DeviceProps::WARP_THREADS,

		// Number of raking threads
		RAKING_THREADS = CtaRakingGrid::RAKING_THREADS,

		// Number of raking elements per warp synchronous raking thread
		RAKING_LENGTH = CtaRakingGrid::RAKING_LENGTH,

		// Cooperative work can be entirely warp synchronous
		WARP_SYNCHRONOUS = (SHARED_ELEMENTS == RAKING_THREADS),
	};


	/**
	 * Warp-scan utility type
	 */
	typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		typename WarpScan::SmemStorage 			warp_scan;		// Buffer for warp-synchronous scan
		typename CtaRakingGrid::SmemStorage 	raking_grid;	// Padded CTA raking grid
	};




	//---------------------------------------------------------------------
	// Exclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int STRIPS,
		int ITEMS,
		typename ScanOp>
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					// (in) SmemStorage reference
		T 				(&input)[STRIPS][ITEMS],		// (in) Input array
		T 				(&output)[STRIPS][ITEMS],		// (out) Output array (may be aliased to input)
		ScanOp 			scan_op,						// (in) Reduction operator
		T				identity,						// (in) Identity value.
		T				&aggregate,						// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)					// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		// Pointers into shared memory raking grid where my strip partial-reductions go
		T *raking_placement[STRIPS];

		// Reduce in registers and place partial into shared memory raking grid
		#pragma unroll
		for (int STRIP = 0; STRIP < STRIPS; STRIP++)
		{
			raking_placement[STRIP]= CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, STRIP);
			*raking_placement[STRIP] = ThreadReduce(input[STRIP], scan_op);
		}

		__syncthreads();

		// Reduce parallelism to one warp
		if (threadIdx.x < RAKING_THREADS)
		{
			// Pointer to my segment in raking grid
			T *raking_segment = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);

			// Raking upsweep reduction
			T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_segment, scan_op);

			// Warp synchronous scan
			T exclusive_partial;
			WarpScan::ExclusiveScan(
				smem_storage.warp_scan,
				raking_partial,
				exclusive_partial,
				scan_op,
				identity,
				aggregate,
				warp_prefix);

			// Raking downsweep scan
			ThreadScanExclusive<RAKING_LENGTH>(raking_segment, raking_segment, scan_op, exclusive_partial);
		}

		__syncthreads();

		// Scan in registers, prefixed by the exclusive partial from shared memory grid
		#pragma unroll
		for (int STRIP = 0; STRIP < STRIPS; STRIP++)
		{
			ThreadScanExclusive(input[STRIP], output[STRIP], scan_op, *raking_placement[STRIP]);
		}

	}

};



} // namespace cub
CUB_NS_POSTFIX
