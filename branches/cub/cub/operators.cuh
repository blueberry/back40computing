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
 * Simple functor operators
 ******************************************************************************/

#pragma once

#include "type_utils.cuh"
#include "ns_wrapper.cuh"


CUB_NS_PREFIX
namespace cub {


/**
 * Cast transform functor
 */
template <typename T, typename S>
struct CastTransformOp
{
	__host__ __device__ __forceinline__ T operator ()(const S &item)
	{
		return (T) item;
	}
};


/**
 * Default equality functor
 */
template <typename T>
struct Equality
{
	__host__ __device__ __forceinline__ bool operator()(const T &a, const T &b)
	{
		return a == b;
	}
};


/**
 * Default sum functor
 */
template <
	typename T,
	bool PRIMITIVE = Traits<T>::PRIMITIVE>
struct Sum
{
	// Binary reduction
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return a + b;
	}
};


/**
 * Default sum functor
 */
template <typename T>
struct Max
{
	// Binary reduction
	__host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
	{
		return CUB_MAX(a, b);
	}
};




} // namespace cub
CUB_NS_POSTFIX
