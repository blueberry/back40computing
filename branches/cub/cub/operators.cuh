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

/**
 * \file
 * Simple binary operator functor types
 */

/******************************************************************************
 * Simple functor operators
 ******************************************************************************/

#pragma once

#include "type_utils.cuh"
#include "ns_wrapper.cuh"


CUB_NS_PREFIX

/// CUB namespace
namespace cub {

/**
 *  \addtogroup SimtUtils
 * @{
 */

/**
 * Default equality functor
 */
template <typename T>
struct Equality
{
    /// Boolean equality operator, returns <tt>(a == b)</tt>
    __host__ __device__ __forceinline__ bool operator()(const T &a, const T &b)
    {
        return a == b;
    }
};


/**
 * Default sum functor
 */
template <typename T>
struct Sum
{
    /// Boolean sum operator, returns <tt>a + b</tt>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
    {
        return a + b;
    }
};


/**
 * Default max functor
 */
template <typename T>
struct Max
{
    /// Boolean max operator, returns <tt>(a > b) ? a : b</tt>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b)
    {
        return CUB_MAX(a, b);
    }
};



/** @} */       // end of SimtUtils group


} // namespace cub
CUB_NS_POSTFIX
