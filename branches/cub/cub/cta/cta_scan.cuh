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
#include "../ns_umbrella.cuh"

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
 * 			input, each thread acquires STRIPS "strips" of ELEMENTS consecutive
 * 			inputs, where the stride between a given thread's strips is
 * 			(ELEMENTS * CTA_THREADS) elements.)
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- The scan type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	int 		CTA_THREADS,			// The CTA size in threads
	typename 	T,						// The scan type
	bool 		RETURN_ALL = false, 	// Whether to return the reduced aggregate in all threads (or just thread-0).
	int 		CTA_STRIPS = 1>			// When strip-mining, the number of CTA-strips per tile
struct CtaScan
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	enum
	{
		// Number of active warps
		WARPS = (CTA_THREADS + DeviceProps::WARP_THREADS - 1) / DeviceProps::WARP_THREADS,

		// The total number of elements that need to be cooperatively reduced
		SHARED_ELEMENTS = CTA_THREADS * CTA_STRIPS,

		// Cooperative work can be entirely warp synchronous
		WARP_SYNCHRONOUS = (SHARED_ELEMENTS <= DeviceProps::WARP_THREADS),
	};

	/**
	 * Layout type for padded CTA raking grid
	 */
	typedef CtaRakingGrid<CTA_THREADS, T, CTA_STRIPS> CtaRakingGrid;

	/**
	 * Warp-scan utility type
	 */
	typedef WarpScan<WARPS, T> WarpScan;

	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		typename WarpScan::SmemStorage 			warp_scan;		// Buffer for warp-synchronous scan
		typename CtaRakingGrid::SmemStorage 	raking_grid;	// Padded CTA raking grid
	};



	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial. The aggregate is
	 * returned in thread-0 (and is undefined for other threads).
	 */
	template <typename ScanOp>						// Reduction operator type
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partials[CTA_STRIPS],		// Calling thread's input partial reductions
		ScanOp 			scan_op)					// Reduction operator
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp synchronous reduction (there
			// is only one strip)
			partials[0] = WarpScan::InclusiveScan(
				smem_storage.warp_scan,
				partials[0],
				scan_op);

			if (RETURN_ALL)
			{
				// Load result from thread-0
				partial = smem_storage.warp_buffer[0];
			}
		}
		else
		{
			// Raking reduction.  Place CTA-strided partials into raking grid.
			#pragma unroll
			for (int STRIP = 0; STRIP < CTA_STRIPS; STRIP++)
			{
				// Place partial into shared memory grid.
				*CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, STRIP) = partials[STRIP];
			}

			__syncthreads();

			// Reduce parallelism to one warp
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking reduction. Compute pointer to raking segment and load first element.
				T *raking_segment 	= CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				partial 			= *raking_segment;

				#pragma unroll
				for (int ELEMENT = 1; ELEMENT < RAKING_LENGTH; ELEMENT++)
				{
					// Determine logical index of partial
					unsigned int logical_partial = (threadIdx.x * RAKING_LENGTH) + ELEMENT;

					if (UNGUARDED || (logical_partial < num_valid))
					{
						partial = reduction_op(partial, raking_segment[ELEMENT]);
					}
				}

				// Warp synchronous reduction
				partial = Iterate<0, WARP_SYNCH_STEPS, UNGUARDED>::WarpReduce(
					smem_storage,
					partial,
					num_valid,
					reduction_op);
			}

			if (RETURN_ALL)
			{
				// Barrier and load result from thread-0
				__syncthreads();
				partial = smem_storage.warp_buffer[0];
			}
		}

		return partial;
	}

	}



	//---------------------------------------------------------------------
	// Scan interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <
		int ELEMENTS,
		typename ScanOp>
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage	&smem_storage,		// SmemStorage reference
		T (&input)[LENGTH],				// Input array
		T (&output)[LENGTH],			// Output array (may be aliased to input)
		ScanOp scan_op)					// Reduction operator
	{
		// Compute thread partial reductions
		T partial = ThreadReduce(input, scan_op);

		if (!WARP_SYNCHRONOUS)
		{
			// Raking upsweep: place partial into grid

		}


		return Reduce(smem_storage, partial, CTA_THREADS, scan_op);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive scan using the specified
	 * scan operator.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <
		int ELEMENTS,
		typename ScanOp>
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage	&smem_storage,		// SmemStorage reference
		T (&data)[ELEMENTS],			// Calling thread's input input
		ScanOp scan_op)		// Reduction operator
	{
		return Reduce(smem_storage, partial, CTA_THREADS, scan_op);
	}



};



} // namespace cub
CUB_NS_POSTFIX
