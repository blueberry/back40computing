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
 * Configuration policy for radix sort upsweep reduction kernel
 ******************************************************************************/

#pragma once

#include <cub/cub.cuh>

namespace back40 {
namespace radix_sort {
namespace upsweep {

/**
 * Radix sort upsweep reduction tuning policy.
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_CURRENT_BIT,			// The bit offset of the current radix digit place
	int 							_CURRENT_PASS,			// The number of previous passes
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_LOG_THREADS,			// The number of threads per CTA
	int 							_LOG_LOAD_VEC_SIZE,		// The number of consecutive keys to process per thread per global load
	int 							_LOG_LOADS_PER_TILE,	// The number of loads to process per thread per tile
	cub::LoadModifier 				_LOAD_MODIFIER,			// Load cache-modifier
	cub::StoreModifier 				_STORE_MODIFIER,		// Store cache-modifier
	bool							_SMEM_8BYTE_BANKS,		// Shared memory bank size
	bool 							_EARLY_EXIT>			// Whether or not to short-circuit passes if the upsweep determines homogoneous digits in the current digit place
struct KernelPolicy
{
	enum {
		RADIX_BITS					= _RADIX_BITS,
		CURRENT_BIT 				= _CURRENT_BIT,
		CURRENT_PASS 				= _CURRENT_PASS,
		MIN_CTA_OCCUPANCY  			= _MIN_CTA_OCCUPANCY,
		LOG_THREADS 				= _LOG_THREADS,
		LOG_LOAD_VEC_SIZE  			= _LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE 			= _LOG_LOADS_PER_TILE,
		SMEM_8BYTE_BANKS			= _SMEM_8BYTE_BANKS,
		EARLY_EXIT					= _EARLY_EXIT,

		THREADS						= 1 << LOG_THREADS,
		LOG_TILE_ELEMENTS			= LOG_THREADS + LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
	};

	static const cub::LoadModifier LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const cub::StoreModifier STORE_MODIFIER 		= _STORE_MODIFIER;
};
	


} // namespace upsweep
} // namespace radix_sort
} // namespace back40

