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
 * Reduction over thread-local array types
 ******************************************************************************/

#pragma once

#include <cub/operators.cuh>
#include <cub/ptx_intrinsics.cuh>
#include <cub/type_utils.cuh>

namespace cub {

/**
 * Namespace for utility/iteration structures
 */
namespace thread_reduce
{
	//---------------------------------------------------------------------
	// Helper functions for vectorizing reduction operations
	//---------------------------------------------------------------------

	// Generic case
	template <typename T, typename ReductionOp>
	__device__ __forceinline__ T VectorReduce(
		T a,
		T b,
		T c,
		ReductionOp reduction_op)
	{
		return reduction_op(a, reduction_op(b, c));
	}

	// Specialization for 32-bit int
	template <>
	__device__ __forceinline__ int VectorReduce<int, Sum<int> >(
		int a,
		int b,
		int c,
		Sum<int> reduction_op)
	{
		return util::IADD3(a, b, c);
	};

	// Specialization for 32-bit uint
	template <>
	__device__ __forceinline__ unsigned int VectorReduce<unsigned int, Sum<unsigned int> >(
		unsigned int a,
		unsigned int b,
		unsigned int c,
		Sum<unsigned int> reduction_op)
	{
		return util::IADD3(a, b, c);
	};

	//---------------------------------------------------------------------
	// Iteration Structures (counting down)
	//---------------------------------------------------------------------

	template <int COUNT, int TOTAL>
	struct Iterate
	{
		template <typename T, int ELEMENTS, typename ReductionOp>
		static __device__ __forceinline__ T ThreadReduce(T (&partials)[ELEMENTS], ReductionOp reduction_op)
		{
			T a = Iterate<COUNT - 2, TOTAL>::ThreadReduce(partials, reduction_op);
			T b = partials[TOTAL - COUNT];
			T c = partials[TOTAL - (COUNT - 1)];

			return VectorReduce(a, b, c, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		template <typename T, int ELEMENTS, typename ReductionOp>
		static __device__ __forceinline__ T ThreadReduce(T (&partials)[ELEMENTS], ReductionOp reduction_op)
		{
			return reduction_op(partials[TOTAL - 2], partials[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		template <typename T, int ELEMENTS, typename ReductionOp>
		static __device__ __forceinline__ T ThreadReduce(T (&partials)[ELEMENTS], ReductionOp reduction_op)
		{
			return partials[TOTAL - 1];
		}
	};
	
} // namespace thread_reduce


/**
 * Serial reduction with the specified operator
 */
template <typename ArrayType, typename ReductionOp>
__device__ __forceinline__ T ThreadReduce(
	ArrayType partials,
	ReductionOp reduction_op)
{
	typedef T LinearArray[SEGMENTS * ELEMENTS];

	return serial_reduction::Iterate<
		SEGMENTS * ELEMENTS,
		SEGMENTS * ELEMENTS>::SerialReduce(
			reinterpret_cast<LinearArray&>(partials),
			reduction_op);

	return thread_reduce::Iterate<
		ELEMENTS,
		ELEMENTS>::ThreadReduce(
			partials,
			reduction_op);
}


/**
 * Serial reduction with the addition operator
 */
template <typename ArrayType, int ELEMENTS>
__device__ __forceinline__ T ThreadReduce(
	T (&partials)[ELEMENTS])
{
	Sum<T> reduction_op;
	return ThreadReduce(partials, reduction_op);
}


/**
 * Serial reduction with the specified operator, seeded with the
 * given exclusive partial
 */
template <typename T, int ELEMENTS, typename ReductionOp>
__device__ __forceinline__ T ThreadReduce(
	T (&partials)[ELEMENTS],
	T exclusive_partial,
	ReductionOp reduction_op)
{
	return reduction_op(
		exclusive_partial,
		ThreadReduce(partials, reduction_op));
}

/**
 * Serial reduction with the addition operator, seeded with the
 * given exclusive partial
 */
template <typename T, int ELEMENTS, typename ReductionOp>
__device__ __forceinline__ T ThreadReduce(
	T (&partials)[ELEMENTS],
	T exclusive_partial)
{
	Sum<T> reduction_op;
	return ThreadReduce(partials, exclusive_partial, reduction_op);
}





} // namespace cub


