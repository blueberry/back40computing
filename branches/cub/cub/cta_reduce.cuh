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

#include <cub/device_props.cuh>
#include <cub/type_utils.cuh>
#include <cub/operators.cuh>
#include <cub/ns_umbrella.cuh>
#include <cub/thread_reduce.cuh>

CUB_NS_PREFIX
namespace cub {

/**
 * Cooperative reduction abstraction for CTAs. The aggregate is returned in
 * thread-0 (and is undefined for other threads).
 *
 * Features:
 * 		- Very efficient (only one synchronization barrier)
 * 		- Zero bank conflicts for most types
 * 		- Supports non-commutative reduction operators
 * 		- Supports partially-full CTAs (i.e., high-order threads having undefined values)
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- Every thread has a valid input
 * 		- The reduction type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	int 		CTA_THREADS,		// The CTA size in threads
	typename 	T,					// The reduction type
	int 		CTA_STRIPS = 1>		// When strip-mining a tile, the number of strips per tile
struct CtaReduce
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	enum
	{
		// The total number of elements that need to be cooperatively reduced
		REDUCTION_ELEMENTS = CTA_THREADS * CTA_STRIPS,

		// Whether or not the reduction type is a built-in primitive
		PRIMITIVE = NumericTraits<T>::PRIMITIVE,

		// Number of warp-synchronous raking threads
		RAKING_THREADS = CUB_MIN(CTA_THREADS, DeviceProps::WARP_THREADS),

		// Number of raking elements per warp synchronous raking thread
		RAKING_ELEMENTS = (REDUCTION_ELEMENTS + RAKING_THREADS - 1) / RAKING_THREADS,

		// Number of bytes per shared memory segment
		SEGMENT_BYTES = DeviceProps::SMEM_BANKS * DeviceProps::SMEM_BANK_BYTES,

		// Number of elements per shared memory segment
		SEGMENT_ELEMENTS = (SEGMENT_BYTES + sizeof(T) - 1) / sizeof(T),

		// Stride in elements between padding blocks (insert a padding block after each), must be a multiple of raking elements
		PADDING_STRIP = CUB_ROUND_UP_NEAREST(SEGMENT_ELEMENTS, RAKING_ELEMENTS),

		// Number of elements per padding block
		PADDING_ELEMENTS = (DeviceProps::SMEM_BANK_BYTES + sizeof(T) - 1) / sizeof(T),

		// Total number of shared memory elements
		SMEM_ELEMENTS = REDUCTION_ELEMENTS + (REDUCTION_ELEMENTS / PADDING_STRIP),

		// Number of warp-synchronous steps
		WARP_SYNCH_STEPS = Log2<CUB_MIN(RAKING_THREADS, REDUCTION_ELEMENTS)>::VALUE,

		// Whether or not the number of reduction elements is a multiple of the warp size
		WARP_MULTIPLE = (REDUCTION_ELEMENTS % DeviceProps::WARP_THREADS == 0),

		// Whether or not the number of reduction elements is a power of two
		POWER_OF_TWO = ((REDUCTION_ELEMENTS & (REDUCTION_ELEMENTS - 1)) == 0)
	};


	/**
	 * Qualified type of T to use for warp-synchronous storage.  For built-in primitive
	 * types, we can use volatile qualifier (and can omit syncthreads when warp-synchronous)
	 */
	typedef typename If<(PRIMITIVE), volatile T, T>::Type WarpT;


	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		WarpT 	warp_buffer[RAKING_THREADS];
		union
		{
			char	raking_bytes[1];
			T		raking_grid[SMEM_ELEMENTS];
		};
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
			SmemStorage 	&smem_storage,
			T 				partial,
			ReductionOp 	reduction_op,
			int 			num_valid)
		{
			const int OFFSET = 1 << (WARP_SYNCH_STEPS - COUNT - 1);

			// Prevent compiler from hoisting variables between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Update partial if addend is in range
			if (UNGUARDED || ((threadIdx.x + OFFSET) * RAKING_ELEMENTS < num_valid))
			{
				T addend = smem_storage.warp_buffer[threadIdx.x + OFFSET];
				partial = reduction_op(partial, addend);
			}

			// Share partial into buffer
			smem_storage.warp_buffer[threadIdx.x] = partial;

			// Recurse
			return Iterate<COUNT + 1, MAX, UNGUARDED>::WarpReduce(
				smem_storage,
				partial,
				reduction_op,
				num_valid);
		}
	};


	// Termination
	template <int MAX, bool UNGUARDED>
	struct Iterate<MAX, MAX, UNGUARDED>
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ T WarpReduce(
			SmemStorage 	&smem_storage,
			T 				partial,
			ReductionOp 	reduction_op,
			int 			num_valid)
		{
			return partial;
		}
	};


	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 * Warp reduction.  The aggregate is returned in thread-0 (and is
	 * undefined for other threads).
	 */
	template <bool UNGUARDED, typename ReductionOp>
	static __device__ __forceinline__ T WarpReduce(
		SmemStorage 	&smem_storage,
		T 				partial,
		ReductionOp 	reduction_op,
		int 			num_valid)
	{
		// Share partial into warp synchronous buffer.
		smem_storage.warp_buffer[threadIdx.x] = partial;

		// Iterate warp reduction steps
		return Iterate<0, WARP_SYNCH_STEPS, UNGUARDED>::WarpReduce(
			smem_storage,
			partial,
			reduction_op,
			num_valid);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial. The aggregate is
	 * returned in thread-0 (and is undefined for other threads).
	 */
	template <
		bool UNGUARDED,
		typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partials[CTA_STRIPS],		// Calling thread's input partial reductions
		int 			num_valid,				// Number of threads containing valid elements (may be less than CTA_THREADS)
		ReductionOp 	reduction_op)				// Reduction operator
	{
		T partial = partials[0];

		if ((REDUCTION_ELEMENTS == CTA_THREADS) &&
			(REDUCTION_ELEMENTS <= RAKING_THREADS))
		{
			// Short-circuit directly to warp synchronous reduction
			partial = WarpReduce<UNGUARDED>(smem_storage, partial, reduction_op, num_valid);
		}
		else
		{
			// Raking reduction.  Place CTA-strided partials into raking grid.
			#pragma unroll
			for (int STRIP = 0; STRIP < CTA_STRIPS; STRIP++)
			{
				// Offset for partial
				unsigned int partial_offset = (STRIP * CTA_THREADS) + threadIdx.x;

				// Incorporating a block of padding partials every shared memory segment
				partial_offset += (partial_offset / PADDING_STRIP) * PADDING_ELEMENTS;

				// Place partial into shared memory grid.
				smem_storage.raking_grid[partial_offset] = partials[STRIP];
			}

			__syncthreads();

			// Reduce parallelism to one warp
			if (threadIdx.x < RAKING_THREADS)
			{
				// Compute pointer to raking segment
				unsigned int raking_begin 			= threadIdx.x * RAKING_ELEMENTS;
				unsigned int raking_begin_bytes 	= raking_begin * sizeof(T);
				unsigned int padding_bytes 			= (raking_begin_bytes / (PADDING_STRIP * sizeof(T))) * PADDING_ELEMENTS * sizeof(T);
				T *raking_segment 					= (T*) (smem_storage.raking_bytes + raking_begin_bytes + padding_bytes);

				// Raking reduction
				partial = *raking_segment;

				#pragma unroll
				for (int ELEMENT = 1; ELEMENT < RAKING_ELEMENTS; ELEMENT++)
				{
					if (UNGUARDED || (raking_begin + ELEMENT < num_valid))
					{
						partial = reduction_op(partial, raking_segment[ELEMENT]);
					}
				}

				// Warp synchronous reduction
				partial = WarpReduce<UNGUARDED>(smem_storage, partial, reduction_op, num_valid);
			}
		}

		return partial;
	}


	//---------------------------------------------------------------------
	// Partial-tile reduction interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial. The aggregate is
	 * returned in thread-0 (and is undefined for other threads).
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partial,					// Calling thread's input partial reduction
		int 			num_valid,				// Number of threads containing valid elements (may be less than CTA_THREADS)
		ReductionOp 	reduction_op)				// Reduction operator
	{
		// Determine if we don't need bounds checking
		if ((POWER_OF_TWO || WARP_MULTIPLE) && (num_valid == CTA_THREADS))
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
	 * threads each contribute one reduction partial. The aggregate is
	 * returned in thread-0 (and is undefined for other threads).
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
	 * Perform a cooperative, CTA-wide reduction using the specified
	 * reduction operator.
	 *
	 * The aggregate is returned in thread-0 (and is undefined for
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
	 * Perform a cooperative, CTA-wide reduction (sum).
	 *
	 * The aggregate is returned in thread-0 (and is undefined for
	 * other threads).
	 */
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,				// SmemStorage reference
		T 				partial)					// Calling thread's input partial reduction
	{
		return Reduce(smem_storage, partial, CTA_THREADS);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction using the specified
	 * reduction operator.
	 *
	 * The aggregate is returned in thread-0 (and is undefined for
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
	 * Perform a cooperative, CTA-wide reduction (sum).
	 *
	 * The aggregate is returned in thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS, typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				(&tile)[ELEMENTS])				// Calling thread's input
	{
		Sum<T> reduction_op;
		return Reduce(smem_storage, tile, reduction_op);
	}


	/**
	 * Perform a cooperative, CTA-wide reduction using the specified
	 * reduction operator.
	 *
	 * The aggregate is returned in thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS, typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				tile[CTA_STRIPS][ELEMENTS],	// Calling thread's input
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
		if (POWER_OF_TWO || WARP_MULTIPLE)
		{
			return Reduce<true>(smem_storage, segment_partials, TILE_SIZE, reduction_op);
		}
		else
		{
			return Reduce<false>(smem_storage, segment_partials, TILE_SIZE, reduction_op);
		}
	}


	/**
	 * Perform a cooperative, CTA-wide reduction (sum).
	 *
	 * The aggregate is returned in thread-0 (and is undefined for
	 * other threads).
	 */
	template <int ELEMENTS, typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		SmemStorage		&smem_storage,					// SmemStorage reference
		T 				tile[CTA_STRIPS][ELEMENTS])	// Calling thread's input
	{
		Sum<T> reduction_op;
		return Reduce(smem_storage, tile, reduction_op);
	}

};


} // namespace cub
CUB_NS_POSTFIX
