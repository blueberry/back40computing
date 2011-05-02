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
 * Simple reduction operators
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * Empty default transform function
 */
template <typename T>
__device__ __forceinline__ void NopTransform(T &val) {}


/**
 * Addition binary associative operator
 */
template <typename T>
__host__ __device__ __forceinline__ T DefaultSum(const T &a, const T &b)
{
	return a + b;
}

/**
 * Identity for binary addition operator
 */
template <typename T>
__host__ __device__ __forceinline__ T DefaultSumIdentity()
{
	return (T) 0;
}


} // namespace util
} // namespace b40c

