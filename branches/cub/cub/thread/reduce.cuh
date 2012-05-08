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
 * Reduction over thread-local array types.
 *
 * For example:
 *
 * 	int a[4] 		= {1, 2, 3, 4};
 * 	Reduce(a));		// 10
 *
 *  int b[2][2] 	= {{1, 2}, {3, 4}};
 * 	Reduce(b));		// 10
 *
 * 	int *c 			= a + 1;
 * 	Reduce(c));		// 2
 * 	Reduce<2>(c));	// 5
 *
 * 	int (*d)[2] 	= b + 1;
 * 	Reduce(d));		// 7
 *
 ******************************************************************************/

#pragma once

#include <cub/operators.cuh>
#include <cub/type_utils.cuh>

namespace cub {


/**
 * Serial reduction with the specified operator and seed
 */
template <
	int LENGTH,
	typename T,
	typename ReductionOp>
__host__ __device__ __forceinline__ T Reduce(
	T* data,
	ReductionOp reduction_op,
	T seed)
{
	#pragma unroll
	for (int i = 0; i < LENGTH; ++i) {
		seed = reduction_op(seed, data[i]);
	}

	return seed;
}


/**
 * Serial reduction with the specified operator
 */
template <
	int LENGTH,
	typename T,
	typename ReductionOp>
__host__ __device__ __forceinline__ T Reduce(
	T* data,
	ReductionOp reduction_op)
{
	T seed = data[0];
	return Reduce<LENGTH - 1>(data + 1, reduction_op, seed);
}


/**
 * Serial reduction with the addition operator and seed
 */
template <
	int LENGTH,
	typename T>
__host__ __device__ __forceinline__ T Reduce(
	T* data,
	T seed)
{
	Sum<T> reduction_op;
	return Reduce<LENGTH>(data, reduction_op, seed);
}


/**
 * Serial reduction with the addition operator
 */
template <
	int LENGTH,
	typename T>
__host__ __device__ __forceinline__ T Reduce(T* data)
{
	Sum<T> reduction_op;
	return Reduce<LENGTH>(data, reduction_op);
}


/**
 * Serial reduction with the specified operator and seed
 */
template <
	typename ArrayType,
	typename ReductionOp,
	typename T>
__host__ __device__ __forceinline__ T Reduce(
	ArrayType &data,
	ReductionOp reduction_op,
	T seed)
{
	T* linear_array = reinterpret_cast<T*>(data);
	return Reduce<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op, seed);
}


/**
 * Serial reduction with the specified operator
 */
template <
	typename ArrayType,
	typename ReductionOp>
__host__ __device__ __forceinline__ typename ArrayTraits<ArrayType>::Type Reduce(
	ArrayType &data,
	ReductionOp reduction_op)
{
	typedef typename ArrayTraits<ArrayType>::Type T;
	T* linear_array = reinterpret_cast<T*>(data);
	return Reduce<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op);
}


/**
 * Serial reduction with the addition operator and seed
 */
template <typename ArrayType, typename T>
__host__ __device__ __forceinline__ T Reduce(
	ArrayType &data,
	T seed)
{
	Sum<T> reduction_op;
	return Reduce(data, reduction_op, seed);
}


/**
 * Serial reduction with the addition operator
 */
template <typename ArrayType>
__host__ __device__ __forceinline__ typename ArrayTraits<ArrayType>::Type Reduce(
	ArrayType &data)
{
	typedef typename ArrayTraits<ArrayType>::Type T;
	Sum<T> reduction_op;
	return Reduce(data, reduction_op);
}



} // namespace cub


