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
 ******************************************************************************/

/******************************************************************************
 *  Memcopy Granularity Configuration
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"

namespace b40c {
namespace memcopy {


/**
 * Memcopy kernel granularity configuration meta-type.  Parameterizations of this
 * type encapsulate our kernel-tuning parameters (i.e., they are reflected via
 * the static fields).
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	typename _T,
	typename _SizeT,
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	CacheModifier _CACHE_MODIFIER,
	bool _WORK_STEALING,
	int _LOG_SCHEDULE_GRANULARITY = _LOG_THREADS + _LOG_LOAD_VEC_SIZE + _LOG_LOADS_PER_TILE>
struct MemcopyKernelConfig
{
	typedef _T										T;
	typedef _SizeT									SizeT;
	static const int CTA_OCCUPANCY  				= _CTA_OCCUPANCY;
	static const CacheModifier CACHE_MODIFIER 		= _CACHE_MODIFIER;
	static const bool WORK_STEALING					= _WORK_STEALING;

	static const int LOG_THREADS 					= _LOG_THREADS;
	static const int THREADS						= 1 << LOG_THREADS;

	static const int LOG_LOAD_VEC_SIZE  			= _LOG_LOAD_VEC_SIZE;
	static const int LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE;

	static const int LOG_LOADS_PER_TILE 			= _LOG_LOADS_PER_TILE;
	static const int LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE;

	static const int LOG_WARPS						= LOG_THREADS - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__);
	static const int WARPS							= 1 << LOG_WARPS;

	static const int LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE;
	static const int TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD;

	static const int LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS;
	static const int TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS;

	static const int LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY;
	static const int SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY;

	static void Print()
	{
		printf("%d, %d, %d, %d, %d, %d, %s, %s, %d",
			sizeof(T),
			sizeof(SizeT),
			CTA_OCCUPANCY,
			LOG_THREADS,
			LOG_LOAD_VEC_SIZE,
			LOG_LOADS_PER_TILE,
			CacheModifierToString(CACHE_MODIFIER),
			(WORK_STEALING) ? "true" : "false",
			LOG_SCHEDULE_GRANULARITY);
	}
};
		

}// namespace memcopy
}// namespace b40c

