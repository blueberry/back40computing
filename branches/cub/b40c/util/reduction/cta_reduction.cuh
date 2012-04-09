/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * Cooperative reduction abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/operators.cuh>
#include <b40c/util/numeric_traits.cuh>

namespace b40c {
namespace util {
namespace reduction {


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
		CUDA_ARCH						= __B40C_CUDA_ARCH__,							// The target architecture
		WARP_THREADS					= B40C_WARP_THREADS(CUDA_ARCH),					// The number of threads per warp
		NON_PRIMITIVE					= NumericTraits<T>::NAN,						// Whether or not the reduction type is a built-in primitive
		POWER_OF_TWO_THREADS			= ((CTA_THREADS & (CTA_THREADS - 1)) == 0)		// Whether nor not the number of CTA threads is a power of two
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
		SmemT reduction_trees[2][CTA_THREADS];
	};


private:

	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	// Shared memory storage needed for reduction
	SmemStorage& smem_storage;


	//---------------------------------------------------------------------
	// Iteration structures
	//---------------------------------------------------------------------

	// General iteration
	template <
		int OFFSET,
		bool WAS_WARPSCAN,
		bool IS_WARPSCAN = (OFFSET <= WARP_THREADS)>
	struct Iterate
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			T partial,
			SmemStorage &smem_storage,
			int num_valid,
			ReductionOp reduction_op,
			bool full_tree)
		{
			const int BUFFER = Log2<OFFSET>::VALUE & 1;		// Alternate buffers between rounds

			// Update partial
			if ((threadIdx.x < OFFSET) &&
				(full_tree || (threadIdx.x + OFFSET < num_valid)))
			{
				T addend = smem_storage.reduction_trees[BUFFER ^ 1][threadIdx.x + OFFSET];
				partial = reduction_op(partial, addend);
			}
			smem_storage.reduction_trees[BUFFER][threadIdx.x] = partial;

			// Barrier between rounds
			__syncthreads();

			// Recurse
			Iterate<OFFSET / 2, WAS_WARPSCAN>::Invoke(
				partial,
				smem_storage,
				num_valid,
				reduction_op,
				full_tree);
		}
	};

	// Transition into warpscan iteration
	template <int OFFSET>
	struct Iterate<OFFSET, false, true>
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			T partial,
			SmemStorage &smem_storage,
			int num_valid,
			ReductionOp reduction_op,
			bool full_tree)
		{
			const int BUFFER = Log2<OFFSET>::VALUE & 1;		// Alternate buffers between rounds

			if (threadIdx.x < OFFSET) {

				// Update my partial
				if (full_tree || (threadIdx.x + OFFSET < num_valid)) {

					T addend = smem_storage.reduction_trees[BUFFER ^ 1][threadIdx.x + OFFSET];
					partial = reduction_op(partial, addend);
				}
				smem_storage.reduction_trees[BUFFER][threadIdx.x] = partial;

				// Prevent compiler from hoisting variables between rounds
				if (NON_PRIMITIVE) __threadfence_block();

				// Recurse in warpscan mode
				Iterate<OFFSET / 2, true>::Invoke(
					partial,
					smem_storage,
					num_valid,
					reduction_op,
					full_tree);
			}
		}
	};

	// Warpscan iteration
	template <int OFFSET, bool WAS_WARPSCAN>
	struct Iterate<OFFSET, WAS_WARPSCAN, true>
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			T partial,
			SmemStorage &smem_storage,
			int num_valid,
			ReductionOp reduction_op,
			bool full_tree)
		{
			const int BUFFER = Log2<OFFSET>::VALUE & 1;		// Alternate buffers between rounds

			// Update my partial
			if (full_tree || (threadIdx.x + OFFSET < num_valid)) {

				T addend = smem_storage.reduction_trees[BUFFER ^ 1][threadIdx.x + OFFSET];
				partial = reduction_op(partial, addend);
			}
			smem_storage.reduction_trees[BUFFER][threadIdx.x] = partial;

			// Prevent compiler from hoisting variables between rounds
			if (NON_PRIMITIVE) __threadfence_block();

			// Recurse in warpscan mode
			Iterate<OFFSET / 2, true>::Invoke(
				partial,
				smem_storage,
				num_valid,
				reduction_op,
				full_tree);

		}
	};

	// Termination
	template <bool WAS_WARPSCAN>
	struct Iterate<0, WAS_WARPSCAN, true>
	{
		template <typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			T partial,
			SmemStorage &smem_storage,
			int num_valid,
			ReductionOp reduction_op,
			bool full_tree)
		{}
	};

public:

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaReduction(SmemStorage &smem_storage) :
		smem_storage(smem_storage)
	{}


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial.
	 */
	template <typename ReductionOp>
	__device__ __forceinline__ T Reduce(
		T partial,								// Calling thread's input partial reduction
		ReductionOp reduction_op,				// Reduction operator
		int num_valid = CTA_THREADS)		// Number of threads containing valid elements (may be less than CTA_THREADS)
	{
		const int ITERATIONS 		= Log2<CTA_THREADS>::VALUE; 		// Number of iterations (log rounded up)
		const int OFFSET 			= 1 << ITERATIONS;					// Right offset (number of active threads) for first round
		const int BUFFER 			= ITERATIONS & 1;

		// Place items into reduction tree
		smem_storage.reduction_trees[BUFFER][threadIdx.x] = partial;

		__syncthreads();

		// Whether or not the reduction tree of valid elements is a full binary tree
		bool full_tree = POWER_OF_TWO_THREADS && (num_valid == CTA_THREADS);

		// Iterate
		Iterate<OFFSET / 2, false>::Invoke(
			partial,
			smem_storage,
			num_valid,
			reduction_op,
			full_tree);

		__syncthreads();

		// Return first thread's stored aggregate
		return smem_storage.reduction_trees[0][0];
	}


	/**
	 * Perform a cooperative, CTA-wide reduction. The first num_valid
	 * threads each contribute one reduction partial.
	 */
	__device__ __forceinline__ T Reduce(
		T partial,								// Calling thread's input partial reduction
		int num_valid = CTA_THREADS)		// Number of threads containing valid elements (may be less than CTA_THREADS)
	{
		Sum<T> reduction_op;
		return Reduce(partial, reduction_op, num_valid);
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c

