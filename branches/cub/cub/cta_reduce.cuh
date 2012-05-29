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
 * Cooperative reduction abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include <cub/cta_raking_grid.cuh>
#include <cub/device_props.cuh>
#include <cub/type_utils.cuh>
#include <cub/operators.cuh>
#include <cub/thread_reduce.cuh>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {

/**
 * Cooperative reduction abstraction for CTAs.
 *
 * Features:
 * 		- Very efficient (only one synchronization barrier).
 * 		- Zero bank conflicts for most types.
 * 		- Supports non-commutative reduction operators.
 * 		- Supports partially-full CTAs (i.e., high-order threads having
 * 			undefined values).
 * 		- Supports reduction over strip-mined CTA tiles.  (For a given tile of
 * 			input, each thread acquires STRIPS "strips" of ELEMENTS consecutive
 * 			inputs, where the stride between a given thread's strips is
 * 			(ELEMENTS * CTA_THREADS) elements.)
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- Every thread has a valid input
 * 		- The reduction type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	int 		CTA_THREADS,			// The CTA size in threads
	typename 	T,						// The reduction type
	bool 		RETURN_ALL = false, 	// Whether to return the reduced aggregate in all threads (or just thread-0).
	int 		CTA_STRIPS = 1>			// When strip-mining, the number of CTA-strips per tile
struct CtaReduce
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	/**
	 * Layout description for padded CTA raking grid
	 */
	typedef CtaRakingGrid<CTA_THREADS, T, CTA_STRIPS> CtaRakingGrid;


	enum
	{
		// Whether or not the reduction type is a built-in primitive
		PRIMITIVE = NumericTraits<T>::PRIMITIVE,

		// The total number of elements that need to be cooperatively reduced
		SHARED_ELEMENTS = CTA_THREADS * CTA_STRIPS,

		// Number of raking threads
		RAKING_THREADS = CtaRakingGrid::RAKING_THREADS,

		// Number of raking elements per warp synchronous raking thread
		RAKING_LENGTH = CtaRakingGrid::RAKING_LENGTH,

		// Number of warp-synchronous steps
		WARP_SYNCH_STEPS = Log2<CUB_MIN(RAKING_THREADS, SHARED_ELEMENTS)>::VALUE,

		// Cooperative work can be entirely warp synchronous
		WARP_SYNCHRONOUS = (SHARED_ELEMENTS == RAKING_THREADS),

		// Whether or not the number of reduction elements is a power of two
		POWER_OF_TWO = ((SHARED_ELEMENTS & (SHARED_ELEMENTS - 1)) == 0),

		// Whether or not we need bounds checking on a full tile (because of odd CTA sizes)
		FULL_UNGUARDED = (POWER_OF_TWO || CtaRakingGrid::FULL_UNGUARDED),
	};


	/**
	 * Qualified type of T to use for warp-synchronous storage.  For
	 * built-in primitive types, we can use volatile qualifier (and can omit
	 * syncthreads when warp-synchronous)
	 */
	typedef typename If<(PRIMITIVE), volatile T, T>::Type WarpT;


	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		WarpT 									warp_buffer[RAKING_THREADS];	// Buffer for warp-synchronous reduction
		typename CtaRakingGrid::SmemStorage 	raking_grid;					// Padded CTA raking grid
	};


	//---------------------------------------------------------------------
	// Iteration structures
	//---------------------------------------------------------------------

	// General iteration
	template <int COUNT, int MAX, bool UNGUARDED>
	struct Iterate
	{
		// WarpReduce
		template <typename ReductionOp>
		static __device__ __forceinline__ T WarpReduce(
			SmemStorage		&smem_storage,				// SmemStorage reference
			T 				partial,					// Calling thread's input partial reduction
			unsigned int 	num_valid,					// Number of threads containing valid elements (may be less than CTA_THREADS)
			ReductionOp 	reduction_op)				// Reduction operator
		{
			const int OFFSET = 1 << COUNT;

			// Share partial into buffer
			smem_storage.warp_buffer[threadIdx.x] = partial;

			// Prevent compiler from hoisting variables between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Update partial if addend is in range
			if (UNGUARDED || ((threadIdx.x + OFFSET) * RAKING_LENGTH < num_valid))
			{
				T addend = smem_storage.warp_buffer[threadIdx.x + OFFSET];
				partial = reduction_op(partial, addend);
			}

			// Recurse
			return Iterate<COUNT + 1, MAX, UNGUARDED>::WarpReduce(
				smem_storage,
				partial,
				num_valid,
				reduction_op);
		}
	};


	// Termination
	template <int MAX, bool UNGUARDED>
	struct Iterate<MAX, MAX, UNGUARDED>
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ T WarpReduce(
			SmemStorage		&smem_storage,				// SmemStorage reference
			T 				partial,					// Calling thread's input partial reduction
			unsigned int 	num_valid,					// Number of threads containing valid elements (may be less than CTA_THREADS)
			ReductionOp 	reduction_op)				// Reduction operator
		{
			if (RETURN_ALL)
			{
				// Share partial into buffer
				smem_storage.warp_buffer[threadIdx.x] = partial;
			}
			return partial;
		}
	};


	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial. The aggregate is
	 * returned in thread-0 (and is undefined for other threads).
	 */
	template <
		bool UNGUARDED,								// Whether we can skip bounds-checking
		typename ReductionOp>						// Reduction operator type
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partials[CTA_STRIPS],		// Calling thread's input partial reductions
		unsigned int 	num_valid,					// Number of threads containing valid elements (may be less than CTA_THREADS)
		ReductionOp 	reduction_op)				// Reduction operator
	{
		T partial = partials[0];

		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp synchronous reduction
			partial = Iterate<0, WARP_SYNCH_STEPS, UNGUARDED>::WarpReduce(
				smem_storage,
				partial,
				num_valid,
				reduction_op);

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


	//---------------------------------------------------------------------
	// Partial-tile reduction interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partial,					// Calling thread's input partial reduction
		int 			num_valid,					// Number of threads containing valid elements (may be less than CTA_THREADS)
		ReductionOp 	reduction_op)				// Reduction operator
	{
		// Determine if we don't need bounds checking
		if (FULL_UNGUARDED && (num_valid == CTA_THREADS))
		{
			return Reduce<true>(smem_storage, &partial, num_valid, reduction_op);
		}
		else
		{
			return Reduce<false>(smem_storage, &partial, num_valid, reduction_op);
		}
	}


	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partial,					// Calling thread's input partial reduction
		int 			num_valid)					// Number of threads containing valid elements (may be less than CTA_THREADS)
	{
		Sum<T> reduction_op;
		return Reduce(smem_storage, partial, num_valid, reduction_op);
	}


	//---------------------------------------------------------------------
	// Full-tile reduction interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction over a full tile using the
	 * specified reduction operator.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				partial,						// Calling thread's input partial reduction
		ReductionOp 	reduction_op)					// Reduction operator
	{
		return Reduce(smem_storage, partial, CTA_THREADS, reduction_op);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction (sum) over a full tile.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partial)					// Calling thread's input partial reduction
	{
		return Reduce(smem_storage, partial, CTA_THREADS);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction over a full tile using the
	 * specified reduction operator.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS, typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				(&tile)[ELEMENTS],				// Calling thread's input
		ReductionOp 	reduction_op)					// Reduction operator
	{
		const int TILE_SIZE = CTA_THREADS * ELEMENTS;

		// Reduce partials
		T partial = ThreadReduce(tile, reduction_op);
		return reduce(smem_storage, partial, TILE_SIZE, reduction_op);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction (sum) over a full tile.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				(&tile)[ELEMENTS])				// Calling thread's input
	{
		Sum<T> reduction_op;
		return Reduce(smem_storage, tile, reduction_op);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction over a full,
	 * strip-mined tile using the specified reduction operator.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS, typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				tile[CTA_STRIPS][ELEMENTS],		// Calling thread's input
		ReductionOp 	reduction_op)					// Reduction operator
	{
		const int TILE_SIZE = CTA_THREADS * CTA_STRIPS * ELEMENTS;

		// Reduce partials within each segment
		T segment_partials[CTA_STRIPS];
		for (int STRIP = 0; STRIP < CTA_STRIPS; STRIP++)
		{
			segment_partials[STRIP] = ThreadReduce(tile[STRIP], reduction_op);
		}

		// Determine if we don't need bounds checking
		if (FULL_UNGUARDED)
		{
			return Reduce<true>(smem_storage, segment_partials, TILE_SIZE, reduction_op);
		}
		else
		{
			return Reduce<false>(smem_storage, segment_partials, TILE_SIZE, reduction_op);
		}
	}


	/**
	 * Perform a cooperative, CTA-wide reduction (sum) over a full,
	 * strip-mined tile.
	 *
	 * If RETURN_ALL, the aggregate is returned in all threads.  Otherwise
	 * the return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				tile[CTA_STRIPS][ELEMENTS])		// Calling thread's input
	{
		Sum<T> reduction_op;
		return Reduce(smem_storage, tile, reduction_op);
	}

};


} // namespace cub
CUB_NS_POSTFIX
