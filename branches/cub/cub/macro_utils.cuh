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
 * Common C/C++ macro utilities
 ******************************************************************************/

#pragma once

#include "ns_wrapper.cuh"

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
 * Quotient of x/y rounded down to nearest integer
 */
#define CUB_QUOTIENT_FLOOR(x, y) ((x) / (y))

/**
 * Quotient of x/y rounded up to nearest integer
 */
#define CUB_QUOTIENT_CEILING(x, y) (((x) + (y) - 1) / (y))

/**
 * x rounded up to the nearest multiple of y
 */
#define CUB_ROUND_UP_NEAREST(x, y) ((((x) + (y) - 1) / (y)) * y)


/**
 * x rounded down to the nearest multiple of y
 */
#define CUB_ROUND_DOWN_NEAREST(x, y) (((x) / (y)) * y)

/**
 * Return character string for given type
 */
#define CUB_TYPE_STRING(type) ""#type



} // namespace cub
CUB_NS_POSTFIX
