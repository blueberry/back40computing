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
 * Reduction over thread-local array types.
 *
 * For example:
 *
 *	Sum<T> op;
 *
 * 	int a[4] = {1, 2, 3, 4};
 * 	ThreadReduce(a, op));						// 10
 *
 *  int b[2][2] = {{1, 2}, {3, 4}};
 * 	ThreadReduce(b, op));						// 10
 *
 * 	int *c = &a[1];
 * 	ThreadReduce(c, op));						// 2
 * 	ThreadReduce<2>(c, op));					// 5
 *
 * 	int (*d)[2] = &b[1];
 * 	ThreadReduce(d, op));						// 7
 *
 ******************************************************************************/

#pragma once

#include "../operators.cuh"
#include "../type_utils.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {

/**
 * Serial reduction with the specified operator and seed
 */
template <
	int LENGTH,
	typename T,
	typename ReductionOp>
__device__ __forceinline__ T ThreadReduce(
	T* data,
	ReductionOp reduction_op,
	T seed)
{
	#pragma unroll
	for (int i = 0; i < LENGTH; ++i)
	{
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
__device__ __forceinline__ T ThreadReduce(
	T* data,
	ReductionOp reduction_op)
{
	T seed = data[0];
	return ThreadReduce<LENGTH - 1>(data + 1, reduction_op, seed);
}


/**
 * Serial reduction with the specified operator and seed
 */
template <
	typename ArrayType,
	typename ReductionOp>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type ThreadReduce(
	ArrayType &data,
	ReductionOp reduction_op,
	typename ArrayTraits<ArrayType>::Type seed)
{
	typedef typename ArrayTraits<ArrayType>::Type T;
	T* linear_array = reinterpret_cast<T*>(data);
	return ThreadReduce<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op, seed);
}


/**
 * Serial reduction with the specified operator
 */
template <
	typename ArrayType,
	typename ReductionOp>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type ThreadReduce(
	ArrayType &data,
	ReductionOp reduction_op)
{
	typedef typename ArrayTraits<ArrayType>::Type T;
	T* linear_array = reinterpret_cast<T*>(data);
	return ThreadReduce<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op);
}


} // namespace cub
CUB_NS_POSTFIX

