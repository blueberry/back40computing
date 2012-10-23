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

#include "../cta/cta_raking_grid.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {

/**
 * Cooperative reduction abstraction for CTAs.
 *
 * Performs a raking upsweep followed by a warp-synchronous Kogge-Stone
 * style reduction.
 *
 * Features:
 * 		- Supports non-commutative reduction operators.
 * 		- Supports partially-full CTAs (i.e., high-order threads having
 * 			undefined values).
 * 		- Very efficient (only one synchronization barrier).
 * 		- Zero bank conflicts for most types.
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- Every thread has a valid input
 * 		- The reduction type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	typename 	T,					// The reduction type
	int 		CTA_THREADS>		// The CTA size in threads
class CtaReduce
{
private:

	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	/**
	 * Layout type for padded CTA raking grid
	 */
	typedef CtaRakingGrid<CTA_THREADS, T, 1> CtaRakingGrid;


	enum
	{
		// Number of raking threads
		RAKING_THREADS = CtaRakingGrid::RAKING_THREADS,

		// Number of raking elements per warp synchronous raking thread
		RAKING_LENGTH = CtaRakingGrid::RAKING_LENGTH,

		// Number of warp-synchronous steps
		WARP_SYNCH_STEPS = Log2<RAKING_THREADS>::VALUE,

		// Cooperative work can be entirely warp synchronous
		WARP_SYNCHRONOUS = (RAKING_THREADS == CTA_THREADS),

		// Whether or not warp-synchronous reduction should be unguarded (i.e., the warp-reduction elements is a power of two
		WARP_SYNCHRONOUS_UNGUARDED = ((RAKING_THREADS & (RAKING_THREADS - 1)) == 0),

		// Whether or not accesses into smem are unguarded
		RAKING_UNGUARDED = CtaRakingGrid::UNGUARDED,

	};

public:

	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		T 										warp_buffer[RAKING_THREADS];	// Buffer for warp-synchronous reduction
		typename CtaRakingGrid::SmemStorage 	raking_grid;					// Padded CTA raking grid
	};

private:

	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 * Warp reduction
	 */
	template <
		bool				FULL_TILE,
		int 				RAKING_LENGTH,
		typename 			ReductionOp>
	static __device__ __forceinline__ T WarpReduce(
		SmemStorage			&smem_storage,			// SmemStorage reference
		T 					partial,				// Calling thread's input partial reduction
		const unsigned int 	&valid_threads,				// Number valid threads (may be less than CTA_THREADS)
		ReductionOp 		reduction_op)			// Reduction operator
	{

		for (int STEP = 0; STEP < WARP_SYNCH_STEPS; STEP++)
		{
			const int OFFSET = 1 << STEP;

			// Share partial into buffer
			ThreadStore<PTX_STORE_VS>(&smem_storage.warp_buffer[threadIdx.x], partial);

			// Update partial if addend is in range
			if ((FULL_TILE && WARP_SYNCHRONOUS_UNGUARDED) || ((threadIdx.x + OFFSET) * RAKING_LENGTH < valid_threads))
			{
				T addend = ThreadLoad<PTX_LOAD_VS>(&smem_storage.warp_buffer[threadIdx.x + OFFSET]);
				partial = reduction_op(partial, addend);
			}
		}

		return partial;
	}



	/**
	 * Perform a cooperative, CTA-wide reduction. The first valid_threads
	 * threads each contribute one reduction partial.
	 *
	 * The return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <
		bool 				FULL_TILE,
		typename			ReductionOp>				// Reduction operator type
	static __device__ __forceinline__ T ReduceHelper(
		SmemStorage			&smem_storage,				// SmemStorage reference
		T 					partial,					// Calling thread's input partial reductions
		const unsigned int 	&valid_threads,				// Number of valid elements (may be less than CTA_THREADS)
		ReductionOp 		reduction_op)				// Reduction operator
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp synchronous reduction (unguarded if active threads is a power-of-two)
			partial = WarpReduce<FULL_TILE, 1>(
				smem_storage,
				partial,
				valid_threads,
				reduction_op);
		}
		else
		{
			// Place partial into shared memory grid.
			*CtaRakingGrid::PlacementPtr(smem_storage.raking_grid) = partial;

			__syncthreads();

			// Reduce parallelism to one warp
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking reduction in grid
				T *raking_segment = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				partial = raking_segment[0];

				#pragma unroll
				for (int ITEM = 1; ITEM < RAKING_LENGTH; ITEM++)
				{
					// Update partial if addend is in range
					if ((FULL_TILE && RAKING_UNGUARDED) || ((threadIdx.x * RAKING_LENGTH) + ITEM < valid_threads))
					{
						partial = reduction_op(partial, raking_segment[ITEM]);
					}
				}

				// Warp synchronous reduction
				partial = WarpReduce<(FULL_TILE && RAKING_UNGUARDED), RAKING_LENGTH>(
					smem_storage,
					partial,
					valid_threads,
					reduction_op);
			}
		}

		return partial;
	}

public:

	//---------------------------------------------------------------------
	// Partial-tile reduction interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first valid_threads
	 * threads each contribute one reduction partial.
	 *
	 * The return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage			&smem_storage,				// SmemStorage reference
		T 					partial,					// Calling thread's input partial reduction
		const unsigned int 	&valid_threads,				// Number of threads containing valid elements (may be less than CTA_THREADS)
		ReductionOp 		reduction_op)				// Reduction operator
	{
		// Determine if we don't need bounds checking
		if (valid_threads == CTA_THREADS)
		{
			return ReduceHelper<true>(smem_storage, partial, valid_threads, reduction_op);
		}
		else
		{
			return ReduceHelper<false>(smem_storage, partial, valid_threads, reduction_op);
		}
	}


	//---------------------------------------------------------------------
	// Full-tile reduction interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction over a full tile using the
	 * specified reduction operator.
	 *
	 * The return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partial,					// Calling thread's input partial reduction
		ReductionOp 	reduction_op)				// Reduction operator
	{
		return Reduce(smem_storage, partial, CTA_THREADS, reduction_op);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction over a full tile using the
	 * specified reduction operator.
	 *
	 * The return value is only valid for thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS, typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				(&tile)[ELEMENTS],			// Calling thread's input
		ReductionOp 	reduction_op)				// Reduction operator
	{
		// Reduce partials
		T partial = ThreadReduce(tile, reduction_op);
		return Reduce(smem_storage, partial, CTA_THREADS, reduction_op);
	}

};


} // namespace cub
CUB_NS_POSTFIX
