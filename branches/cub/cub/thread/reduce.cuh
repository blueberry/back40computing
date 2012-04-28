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
 * /
namespace reduce
{
	//---------------------------------------------------------------------
	// Helper functions for vectorizing reduction operations
	//---------------------------------------------------------------------

	// Generic case
	template <typename T, typename ReductionOp>
	__device__ __forceinline__ T VectorTReduce(
		T a,
		T b,
		T c,
		ReductionOp reduction_op)
	{
		return reduction_op(a, reduction_op(b, c));
	}

	// Specialization for 32-bit int
	template <>
	__device__ __forceinline__ int VectorTReduce<int, Sum<int> >(
		int a,
		int b,
		int c,
		Sum<int> reduction_op)
	{
		return util::IADD3(a, b, c);
	};

	// Specialization for 32-bit uint
	template <>
	__device__ __forceinline__ unsigned int VectorTReduce<unsigned int, Sum<unsigned int> >(
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
		static __device__ __forceinline__ T Reduce(T (&data)[ELEMENTS], ReductionOp reduction_op)
		{
			T a = Iterate<COUNT - 2, TOTAL>::Reduce(data, reduction_op);
			T b = data[TOTAL - COUNT];
			T c = data[TOTAL - (COUNT - 1)];

			return VectorTReduce(a, b, c, reduction_op);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		template <typename T, int ELEMENTS, typename ReductionOp>
		static __device__ __forceinline__ T Reduce(T (&data)[ELEMENTS], ReductionOp reduction_op)
		{
			return reduction_op(data[TOTAL - 2], data[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		template <typename T, int ELEMENTS, typename ReductionOp>
		static __device__ __forceinline__ T Reduce(T (&data)[ELEMENTS], ReductionOp reduction_op)
		{
			return data[TOTAL - 1];
		}
	};
	
} // namespace reduce
*/

/**
 * Serial reduction with the specified operator
 */
template <typename ArrayType, typename ReductionOp>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type Reduce(
	ArrayType data,
	ReductionOp reduction_op)
{
	return 0;
/*
	ArrayTraits<ArrayType>::

	typedef T LinearArray[SEGMENTS * ELEMENTS];

	return serial_reduction::Iterate<
		SEGMENTS * ELEMENTS,
		SEGMENTS * ELEMENTS>::SerialReduce(
			reinterpret_cast<LinearArray&>(data),
			reduction_op);

	return reduce::Iterate<
		ELEMENTS,
		ELEMENTS>::Reduce(
			data,
			reduction_op);
*/
}


/**
 * Serial reduction with the addition operator
 */
template <typename ArrayType>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type Reduce(
	ArrayType data,
	int num_elements = ArrayTraits<ArrayType>::ELEMENTS)
{
	return 0;
/*
	Sum<T> reduction_op;
	return Reduce(data, reduction_op);
*/
}


/**
 * Serial reduction with the specified operator, seeded with the
 * given exclusive partial
 */
template <typename ArrayType, typename T, typename ReductionOp>
__device__ __forceinline__ T Reduce(
	ArrayType data,
	T seed,
	ReductionOp reduction_op,
	int num_elements = ArrayTraits<ArrayType>::ELEMENTS)
{
	return 0;
/*
	return reduction_op(
		seed,
		Reduce(data, reduction_op));
*/
}

/**
 * Serial reduction with the addition operator, seeded with the
 * given exclusive partial
 */
template <typename ArrayType, typename T>
__device__ __forceinline__ T Reduce(
	ArrayType data,
	T seed,
	int num_elements = ArrayTraits<ArrayType>::ELEMENTS)
{
	return 0;
/*
	Sum<T> reduction_op;
	return Reduce(data, seed, reduction_op);
*/
}





} // namespace cub


