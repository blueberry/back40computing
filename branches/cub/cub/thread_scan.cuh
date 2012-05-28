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
 * Scan over thread-local array types
 ******************************************************************************/

#pragma once

#include <cub/operators.cuh>
#include <cub/ptx_intrinsics.cuh>
#include <cub/type_utils.cuh>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {


/******************************************************************************
 * Exclusive scan
 ******************************************************************************/

/**
 * Serial exclusive scan with the specified operator and seed
 */
template <
	int LENGTH,
	typename T,
	typename ReductionOp>
__device__ __forceinline__ T ThreadScanExcl(
	T* data,
	ReductionOp reduction_op,
	T seed)
{
	T exclusive_partial = seed;
	T inclusive_partial = seed;

	#pragma unroll
	for (int i = 0; i < LENGTH; ++i)
	{
		T inclusive_partial = reduction_op(exclusive_partial, data[i]);
		data[i] = exclusive_partial;
		exclusive_partial = inclusive_partial;
	}

	return inclusive_partial;
}


/**
 * Serial exclusive scan with the addition operator and seed
 */
template <
	int LENGTH,
	typename T>
__device__ __forceinline__ T ThreadScanExcl(
	T* data,
	T seed)
{
	Sum<T> reduction_op;
	return ThreadScanExcl<LENGTH>(data, reduction_op, seed);
}


/**
 * Serial exclusive scan with the specified operator and seed
 */
template <
	typename ArrayType,
	typename ReductionOp,
	typename T>
__device__ __forceinline__ T ThreadScanExcl(
	ArrayType &data,
	ReductionOp reduction_op,
	T seed)
{
	T* linear_array = reinterpret_cast<T*>(data);
	return ThreadScanExcl<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op, seed);
}


/**
 * Serial exclusive scan with the addition operator and seed
 */
template <typename ArrayType, typename T>
__device__ __forceinline__ T ThreadScanExcl(
	ArrayType &data,
	T seed)
{
	Sum<T> reduction_op;
	return ThreadScanExcl(data, reduction_op, seed);
}


/******************************************************************************
 * Inclusive scan
 ******************************************************************************/

/**
 * Serial inclusive scan with the specified operator and seed
 */
template <
	int LENGTH,
	typename T,
	typename ReductionOp>
__device__ __forceinline__ T ThreadScanIncl(
	T* data,
	ReductionOp reduction_op,
	T seed)
{
	#pragma unroll
	for (int i = 0; i < LENGTH; ++i) {
		seed = reduction_op(seed, data[i]);
		data[i] = seed;
	}

	return seed;
}


/**
 * Serial inclusive scan with the specified operator
 */
template <
	int LENGTH,
	typename T,
	typename ReductionOp>
__device__ __forceinline__ T ThreadScanIncl(
	T* data,
	ReductionOp reduction_op)
{
	T seed = data[0];
	return ThreadScanIncl<LENGTH - 1>(data + 1, reduction_op, seed);
}


/**
 * Serial inclusive scan with the addition operator and seed
 */
template <
	int LENGTH,
	typename T>
__device__ __forceinline__ T ThreadScanIncl(
	T* data,
	T seed)
{
	Sum<T> reduction_op;
	return ThreadScanIncl<LENGTH>(data, reduction_op, seed);
}


/**
 * Serial inclusive scan with the addition operator
 */
template <
	int LENGTH,
	typename T>
__device__ __forceinline__ T ThreadScanIncl(T* data)
{
	Sum<T> reduction_op;
	return ThreadScanIncl<LENGTH>(data, reduction_op);
}


/**
 * Serial inclusive scan with the specified operator and seed
 */
template <
	typename ArrayType,
	typename ReductionOp,
	typename T>
__device__ __forceinline__ T ThreadScanIncl(
	ArrayType &data,
	ReductionOp reduction_op,
	T seed)
{
	T* linear_array = reinterpret_cast<T*>(data);
	return ThreadScanIncl<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op, seed);
}


/**
 * Serial inclusive scan with the specified operator
 */
template <
	typename ArrayType,
	typename ReductionOp>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type ThreadScanIncl(
	ArrayType &data,
	ReductionOp reduction_op)
{
	typedef typename ArrayTraits<ArrayType>::Type T;
	T* linear_array = reinterpret_cast<T*>(data);
	return ThreadScanIncl<ArrayTraits<ArrayType>::ELEMENTS>(linear_array, reduction_op);
}


/**
 * Serial inclusive scan with the addition operator and seed
 */
template <typename ArrayType, typename T>
__device__ __forceinline__ T ThreadScanIncl(
	ArrayType &data,
	T seed)
{
	Sum<T> reduction_op;
	return ThreadScanIncl(data, reduction_op, seed);
}


/**
 * Serial inclusive scan with the addition operator
 */
template <typename ArrayType>
__device__ __forceinline__ typename ArrayTraits<ArrayType>::Type ThreadScanIncl(
	ArrayType &data)
{
	typedef typename ArrayTraits<ArrayType>::Type T;
	Sum<T> reduction_op;
	return ThreadScanIncl(data, reduction_op);
}


} // namespace cub
CUB_NS_POSTFIX

