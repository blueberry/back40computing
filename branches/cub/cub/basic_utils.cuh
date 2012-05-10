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
 * Basic, common utility subroutines
 ******************************************************************************/

#pragma once

#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {


/**
 * Select maximum(a, b)
 */
#define CUB_MAX(a, b) (((a) > (b)) ? (a) : (b))


/**
 * Select minimum(a, b)
 */
#define CUB_MIN(a, b) (((a) < (b)) ? (a) : (b))


/**
 * x rounded up to the nearest multiple of y
 */
#define CUB_ROUND_UP_NEAREST(x, y) ((((x) + (y) - 1) / (y)) * y)


/**
 * x rounded down to the nearest multiple of y
 */
#define CUB_ROUND_DOWN_NEAREST(x, y) (((x) / (y)) * y)


/**
 * Perform a swap
 */
template <typename T> 
__host__ __device__ __forceinline__ void Swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


/**
 * Allows you to shift by magnitude (left for positive, right for negative).
 *
 * For example:
 *     Shift(8, -2)		// 2
 */
__host__ __device__ __forceinline__ int Shift(int val, const int magnitude)
{
	if (magnitude > 0) {
		return val << magnitude;
	} else {
		return val >> magnitude;
	}
}


} // namespace cub
CUB_NS_POSTFIX
