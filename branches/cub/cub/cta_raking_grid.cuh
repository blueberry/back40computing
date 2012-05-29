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
 * CTA raking grid abstraction.
 *
 * CTA threads place elements into shared "grid" and then reduce
 * parallelism to one "raking" warp whose threads can perform sequential
 * aggregation operations on consecutive sequences of shared items.  Padding
 * is provided to eliminate bank conflicts (for most data types).
 ******************************************************************************/

#pragma once

#include <cub/device_props.cuh>
#include <cub/type_utils.cuh>
#include <cub/ns_umbrella.cuh>

CUB_NS_PREFIX
namespace cub {


/**
 * CTA raking grid abstraction.
 *
 * CTA threads place elements into shared "grid" and then reduce
 * parallelism to one "raking" warp whose threads can perform sequential
 * aggregation operations on consecutive sequences of shared items.  Padding
 * is provided to eliminate bank conflicts (for most data types).
 */
template <
	int 		CTA_THREADS,		// The CTA size in threads
	typename 	T,					// The reduction type
	int 		CTA_STRIPS = 1>		// When strip-mining, the number of CTA-strips per tile
struct CtaRakingGrid
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

	enum
	{
		// The total number of elements that need to be cooperatively reduced
		SHARED_ELEMENTS = CTA_THREADS * CTA_STRIPS,

		// Number of warp-synchronous raking threads
		RAKING_THREADS = CUB_MIN(CTA_THREADS, DeviceProps::WARP_THREADS),

		// Number of raking elements per warp-synchronous raking thread (rounded up)
		RAKING_LENGTH = (SHARED_ELEMENTS + RAKING_THREADS - 1) / RAKING_THREADS,

		// Total number of raking elements in the grid
		RAKING_ELEMENTS = RAKING_THREADS * RAKING_LENGTH,

		// Number of bytes per shared memory segment
		SEGMENT_BYTES = DeviceProps::SMEM_BANKS * DeviceProps::SMEM_BANK_BYTES,

		// Number of elements per shared memory segment (rounded up)
		SEGMENT_LENGTH = (SEGMENT_BYTES + sizeof(T) - 1) / sizeof(T),

		// Stride in elements between padding blocks (insert a padding block after each), must be a multiple of raking elements
		PADDING_STRIDE = CUB_ROUND_UP_NEAREST(SEGMENT_LENGTH, RAKING_LENGTH),

		// Number of elements per padding block
		PADDING_ELEMENTS = (DeviceProps::SMEM_BANK_BYTES + sizeof(T) - 1) / sizeof(T),

		// Total number of elements in the raking grid
		GRID_ELEMENTS = RAKING_ELEMENTS + (RAKING_ELEMENTS / PADDING_STRIDE),

		// Whether or not we may need bounds checking on a full tile (the number
		// of reduction elements is not a multiple of the warp size)
		FULL_UNGUARDED = (SHARED_ELEMENTS % DeviceProps::WARP_THREADS == 0),
	};


	/**
	 * Shared memory storage type
	 */
	typedef T SmemStorage[CtaRakingGrid::GRID_ELEMENTS];


	/**
	 * Pointer for placement into raking grid (with padding)
	 */
	static __device__ __forceinline__ T* PlacementPtr(
		SmemStorage &smem_storage,
		int cta_strip = 0)
	{
		// Offset for partial
		unsigned int offset = (cta_strip * CTA_THREADS) + threadIdx.x;

		// Incorporating a block of padding partials every shared memory segment
		return smem_storage + offset + (offset / PADDING_STRIDE) * PADDING_ELEMENTS;
	}


	/**
	 * Pointer for sequential warp-synchronous raking within grid (with padding)
	 */
	static __device__ __forceinline__ T* RakingPtr(SmemStorage &smem_storage)
	{
		unsigned int raking_begin_bytes 	= threadIdx.x * RAKING_LENGTH * sizeof(T);
		unsigned int padding_bytes 			= (raking_begin_bytes / (PADDING_STRIDE * sizeof(T))) * PADDING_ELEMENTS * sizeof(T);

		return reinterpret_cast<T*>(
			reinterpret_cast<char*>(smem_storage) +
			raking_begin_bytes +
			padding_bytes);
	}
};



} // namespace cub
CUB_NS_POSTFIX
