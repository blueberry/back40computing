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
 * Kernel utilities for initializing 2D arrays (tiles)
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {
namespace io {


/**
 * Initialize a tile of items
 */
template <
	int LOG_LOADS_PER_TILE,
	int LOG_LOAD_VEC_SIZE>
struct InitializeTile
{
	static const int LOADS_PER_TILE = 1 << LOG_LOADS_PER_TILE;
	static const int LOAD_VEC_SIZE = 1 << LOG_LOAD_VEC_SIZE;

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	// Iterate over vec-elements
	template <int LOAD, int VEC>
	struct Iterate
	{
		template <typename T, typename S>
		static __device__ __forceinline__ void Copy(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE])
		{
			target[LOAD][VEC] = source[LOAD][VEC];
			Iterate<LOAD, VEC + 1>::Copy(target, source);
		}

		template <typename T, typename S>
		static __device__ __forceinline__ void Init(
			T target[][LOAD_VEC_SIZE],
			const S datum)
		{
			target[LOAD][VEC] = datum;
			Iterate<LOAD, VEC + 1>::Init(target, datum);
		}
	};

	// Iterate over loads
	template <int LOAD>
	struct Iterate<LOAD, LOAD_VEC_SIZE>
	{
		template <typename T, typename S>
		static __device__ __forceinline__ void Copy(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE])
		{
			Iterate<LOAD + 1, 0>::Copy(target, source);
		}

		template <typename T, typename S>
		static __device__ __forceinline__ void Init(
			T target[][LOAD_VEC_SIZE],
			const S datum)
		{
			Iterate<LOAD + 1, 0>::Init(target, datum);
		}
	};

	// Terminate
	template <int VEC>
	struct Iterate<LOADS_PER_TILE, VEC>
	{
		template <typename T, typename S>
		static __device__ __forceinline__ void Copy(
			T target[][LOAD_VEC_SIZE],
			S source[][LOAD_VEC_SIZE]) {}

		template <typename T, typename S>
		static __device__ __forceinline__ void Init(
			T target[][LOAD_VEC_SIZE],
			const S datum) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Copy source to target
	 */
	template <typename T, typename S>
	static __device__ __forceinline__ void Copy(
		T target[][LOAD_VEC_SIZE],
		S source[][LOAD_VEC_SIZE])
	{
		Iterate<0, 0>::Copy(target, source);
	}


	/**
	 * Initialize target with datum
	 */
	template <typename T, typename S>
	static __device__ __forceinline__ void Init(
		T target[][LOAD_VEC_SIZE],
		const S datum)
	{
		Iterate<0, 0>::Init(target, datum);
	}
};


} // namespace io
} // namespace util
} // namespace b40c

