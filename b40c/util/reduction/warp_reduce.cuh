/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * Cooperative warp-reduction
 *
 * Does not support non-commutative operators.  (Suggested to use a warpscan
 * instead for those scenarios
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Perform NUM_ELEMENTS of warp-synchronous reduction.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * Can be used to perform concurrent, independent warp-reductions if
 * storage pointers and their local-thread indexing id's are set up properly.
 *
 * Requires a 2D "warpscan" structure of smem storage having dimensions [2][NUM_ELEMENTS].
 */
template <int LOG_NUM_ELEMENTS>				// Log of number of elements to warp-reduce
struct WarpReduce
{
	enum {
		NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS,
	};


	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T exclusive_partial,
			volatile T warpscan[][NUM_ELEMENTS],
			ReductionOp reduction_op,
			int warpscan_tid)
		{
			// Store exclusive partial
			warpscan[1][warpscan_tid] = exclusive_partial;

			// Load current partial
			T current_partial = warpscan[1][warpscan_tid - OFFSET_LEFT];

			// Compute inclusive partial
			T inclusive_partial = reduction_op(exclusive_partial, current_partial);

			// Recurse
			return Iterate<OFFSET_LEFT / 2>::Invoke(
				inclusive_partial, warpscan, reduction_op, warpscan_tid);
		}
	};
	
	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		template <typename T, typename ReductionOp>
		static __device__ __forceinline__ T Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS],
			ReductionOp reduction_op,
			int warpscan_tid)
		{
			return partial;
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Warp reduction with the specified operator, result is returned in last warpscan thread
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T InvokeSingle(
		T partial,								// Input partial
		volatile T warpscan[][NUM_ELEMENTS],	// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		ReductionOp reduction_op,				// Reduction operator
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Iterate<NUM_ELEMENTS / 2>::Invoke(
			partial, warpscan, warpscan_tid);
	}


	/**
	 * Warp reduction with the addition operator, result is returned in last warpscan thread
	 */
	template <typename T>
	static __device__ __forceinline__ T InvokeSingle(
		T partial,								// Input partial
		volatile T warpscan[][NUM_ELEMENTS],	// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return InvokeSingle(partial, warpscan, Operators<T>::Sum, warpscan_tid);
	}


	/**
	 * Warp reduction with the specified operator, result is returned in all warpscan threads)
	 */
	template <typename T, typename ReductionOp>
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		volatile T warpscan[][NUM_ELEMENTS],	// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		ReductionOp reduction_op,				// Reduction operator
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		T inclusive_partial = InvokeSingle<T, ReductionOp>(
			current_partial, warpscan, warpscan_tid);

		// Write our inclusive partial
		warpscan[1][warpscan_tid] = inclusive_partial;

		// Return last thread's inclusive partial
		return warpscan[1][NUM_ELEMENTS - 1];
	}


	/**
	 * Warp reduction with the addition operator, result is returned in all warpscan threads)
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(
		T current_partial,						// Input partial
		volatile T warpscan[][NUM_ELEMENTS],	// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS
		int warpscan_tid = threadIdx.x)			// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Invoke(current_partial, warpscan, Operators<T>::Sum, warpscan_tid);
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c

