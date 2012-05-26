/******************************************************************************
 * 
 * Copyright (c) 2011-2012, Duane Merrill.  All rights reserved.
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

CUB_NS_PREFIX
namespace cub {

/**
 * Cooperative reduction abstraction for CTAs.
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a power-of-two
 * 		- Every threads has a valid input
 * 		- The reduction type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	int CTA_THREADS,		// The CTA size in threads
	typename T>				// The reduction type
class CtaReduction
{
private:

	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	enum {
		ITERATIONS 						= Log2<CTA_THREADS>::VALUE,				 		// Number of iterations (log rounded up)
		BUFFER_ELEMENTS					= 1 << ITERATIONS,
		FINAL_BUFFER_IDX 				= ITERATIONS & 1,
		PRIMITIVE						= NumericTraits<T>::PRIMITIVE,					// Whether or not the reduction type is a built-in primitive
		WARP_THREADS					= CUB_WARP_THREADS(PTX_ARCH),					// The number of threads per warp
	};

	// Qualified type of T to use for smem storage
	typedef typename If<(NON_PRIMITIVE),
		T, 										// We can't use volatile (and must use syncthreads when warp-synchronous)
		volatile T>::Type SmemT;				// We can use volatile (and can omit syncthreads when warp-synchronous)

public:

	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	// Shared memory storage type (double buffered to require only one
	// barrier per reduction step)
	struct SmemStorage
	{
		SmemT buffers[2][BUFFER_ELEMENTS];
	};


private:

	//---------------------------------------------------------------------
	// Iteration structures
	//---------------------------------------------------------------------

	// General iteration
	template <int COUNT, int MAX>
	struct Iterate
	{
		enum {
			OFFSET			= 1 << COUNT,
			BUFFER_IDX 		= COUNT & 1,						// Alternate buffers between rounds
			IS_WARPSCAN 	= (OFFSET <= WARP_THREADS),
			WAS_WARPSCAN	= (OFFSET * 2 <= WARP_THREADS)
		};

		template <typename ReductionOp>
		static __device__ __forceinline__ void Reduce(
			T partial,
			SmemStorage &smem_storage,
			ReductionOp reduction_op,
			int valid_threads)
		{
			// Share partial into buffer
			smem_storage.buffers[BUFFER_IDX][threadIdx.x] = partial;

			if (IS_WARPSCAN)
			{
				// Prevent compiler from hoisting variables between rounds
				if (!PRIMITIVE) __threadfence_block();
			}
			else
			{
				// Not warp-synchronous: barrier between rounds
				__syncthreads();
			}

			// Update partial if in range
			if ((valid_threads == BUFFER_ELEMENTS) || (threadIdx.x + OFFSET < valid_threads))
			{
				T addend = smem_storage.buffers[BUFFER_IDX][threadIdx.x + OFFSET];
				partial = reduction_op(partial, addend);
			}

			// Recurse
			if ((!IS_WARPSCAN) || WAS_WARPSCAN || (threadIdx.x < WARP_THREADS))
			{
				Iterate<COUNT + 1, MAX>::Unguarded(partial, smem_storage, reduction_op);
			}
		}
	};


	// Termination
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ void Reduce(
			T partial,
			SmemStorage &smem_storage,
			ReductionOp reduction_op,
			int valid_threads)
		{
			// Share partial into buffer
			smem_storage.buffers[FINAL_BUFFER_IDX][threadIdx.x] = partial;

			// Prevent compiler from hoisting variables between rounds
			if (!PRIMITIVE) __threadfence_block();
		}
	};

public:

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ T Reduce(
		T partial,								// Calling thread's input partial reduction
		ReductionOp reduction_op,				// Reduction operator
		int valid_threads = CTA_THREADS)		// Number of threads containing valid elements (may be less than CTA_THREADS)
	{
		// Iterate
		Iterate<0, ITERATIONS>::Invoke(
			partial,
			smem_storage,
			reduction_op,
			valid_threads);

		// Return first thread's stored aggregate
		return smem_storage.buffers[FINAL_BUFFER_IDX][0];
	}


	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial.
	 */
	__device__ __forceinline__ T Reduce(
		T partial,								// Calling thread's input partial reduction
		int num_valid = CTA_THREADS)			// Number of threads containing valid elements (may be less than CTA_THREADS)
	{
		Sum<T> reduction_op;
		return Reduce(partial, reduction_op, num_valid);
	}
};


} // namespace cub
CUB_NS_POSTFIX
