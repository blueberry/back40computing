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
 * Types and subroutines utilities that are common across all B40C sorting 
 * kernels and host enactors  
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace sort {

/**
 * Value-type structure denoting keys-only sorting
 */
struct KeysOnly {};

/**
 * Returns whether or not the templated type indicates keys-only sorting
 */
template <typename V>
__forceinline__ __host__ __device__ bool IsKeysOnly() 
{
	return util::Equals<V, KeysOnly>::VALUE;
}


} // namespace sort
} // namespace b40c

