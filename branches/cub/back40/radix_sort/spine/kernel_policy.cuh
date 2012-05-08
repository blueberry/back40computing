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
 * Configuration policy for radix sort spine scan kernel
 ******************************************************************************/

#pragma once

#include <cub/cub.cuh>

namespace back40 {
namespace radix_sort {
namespace spine {


/**
 * Spine tuning policy.
 */
template <
	int 							_LOG_THREADS,			// The number of threads per CTA
	int 							_LOG_LOAD_VEC_SIZE,		// The number of consecutive keys to process per thread per global load
	int 							_LOG_LOADS_PER_TILE,	// The number of loads to process per thread per tile
	cub::LoadModifier 	_LOAD_MODIFIER,			// Load cache-modifier
	cub::StoreModifier 	_STORE_MODIFIER>		// Store cache-modifier
struct KernelPolicy
{
	enum {
		LOG_THREADS 				= _LOG_THREADS,
		LOG_LOAD_VEC_SIZE  			= _LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE 			= _LOG_LOADS_PER_TILE,

		THREADS						= 1 << LOG_THREADS,
		LOG_TILE_ELEMENTS			= LOG_THREADS + LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
	};

	static const cub::LoadModifier LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const cub::StoreModifier STORE_MODIFIER 	= _STORE_MODIFIER;
};


} // namespace spine
} // namespace radix_sort
} // namespace back40

