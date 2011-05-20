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
 * SerialReduce
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>

namespace b40c {
namespace util {
namespace reduction {

/**
 * Have each thread concurrently perform a serial reduction over its specified segment
 */
template <int NUM_ELEMENTS>
struct SerialReduce
{
	//---------------------------------------------------------------------
	// Iteration Structures
	//---------------------------------------------------------------------

	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate
	{
		template <typename T, T ReductionOp(const T&, const T&)>
		static __device__ __forceinline__ T Invoke(T *partials)
		{
			T a = Iterate<COUNT - 2, TOTAL>::template Invoke<T, ReductionOp>(partials);
			T b = partials[TOTAL - COUNT];
			T c = partials[TOTAL - (COUNT - 1)];

			// TODO: consider specializing with a video 3-op instructions on SM2.0+, e.g., asm("vadd.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(a) : "r"(a), "r"(b), "r"(c));
			return ReductionOp(a, ReductionOp(b, c));
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		template <typename T, T ReductionOp(const T&, const T&)>
		static __device__ __forceinline__ T Invoke(T *partials)
		{
			return ReductionOp(partials[TOTAL - 2], partials[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		template <typename T, T ReductionOp(const T&, const T&)>
		static __device__ __forceinline__ T Invoke(T *partials)
		{
			return partials[TOTAL - 1];
		}
	};
	
	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Serial reduction with the specified operator
	 */
	template <
		typename T,
		T ReductionOp(const T&, const T&)>
	static __device__ __forceinline__ T Invoke(T *partials)
	{
		return Iterate<NUM_ELEMENTS, NUM_ELEMENTS>::template Invoke<T, ReductionOp>(partials);
	}

	/**
	 * Serial reduction with the addition operator
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(T *partials)
	{
		return Invoke<T, Operators<T>::Sum>(partials);
	}

	/**
	 * Serial reduction with the specified operator, seeded with the
	 * given exclusive partial
	 */
	template <typename T,
		T ReductionOp(const T&, const T&)>
	static __device__ __forceinline__ T Invoke(T *partials, T exclusive_partial)
	{
		return ReductionOp(
			exclusive_partial,
			Invoke<T, ReductionOp>(partials));
	}

	/**
	 * Serial reduction with the addition operator, seeded with the
	 * given exclusive partial
	 */
	template <typename T>
	static __device__ __forceinline__ T Invoke(T *partials, T exclusive_partial)
	{
		return ReductionOp(
			exclusive_partial,
			Invoke(partials));
	}
};


} // namespace reduction
} // namespace util
} // namespace b40c


