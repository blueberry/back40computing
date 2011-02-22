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
 * Default (i.e., large-problem) "granularity tuning types" for memcopy
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"
#include "memcopy_api_granularity.cuh"

namespace b40c {
namespace memcopy {
namespace large_problem_tuning {

/**
 * Enumeration of architecture-families that we have tuned for
 */
enum Family
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Classifies a given CUDA_ARCH into an architecture-family
 */
template <int CUDA_ARCH>
struct FamilyClassifier
{
	static const Family FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
									(CUDA_ARCH < SM20) ? 	SM13 :
															SM20;
};


/**
 * Granularity parameterization type
 *
 * We can tune this type per SM-architecture, per problem type.
 */
template <int CUDA_ARCH>
struct TunedConfig : TunedConfig<FamilyClassifier<CUDA_ARCH>::FAMILY> {};



//-----------------------------------------------------------------------------
// SM2.0 default granularity parameterization type
//-----------------------------------------------------------------------------

template <>
struct TunedConfig<SM20>
	: MemcopyConfig<
		unsigned int,			// Data type					Use int32s as primary movement type
		8,						// CTA_OCCUPANCY: 				8 CTAs/SM
		7,						// LOG_THREADS: 				128 threads/CTA
		1,						// LOG_LOAD_VEC_SIZE: 			vec-4
		1,						// LOG_LOADS_PER_TILE: 			4 loads
		CG,						// CACHE_MODIFIER: 				CG (cache global only)
		true,					// WORK_STEALING: 				Work-stealing load-balancing
		9						// LOG_SCHEDULE_GRANULARITY:	2048 items
	> {};



//-----------------------------------------------------------------------------
// SM1.3 default granularity parameterization type
//-----------------------------------------------------------------------------

template <>
struct TunedConfig<SM13>
	: MemcopyConfig<
		unsigned short,			// Data type					Use int16s as primary movement type
		8,						// CTA_OCCUPANCY: 				8 CTAs/SM
		7,						// LOG_THREADS: 				128 threads/CTA
		2,						// LOG_LOAD_VEC_SIZE: 			vec-4
		0,						// LOG_LOADS_PER_TILE: 			4 loads
		NONE,					// CACHE_MODIFIER: 				CA (cache all levels)
		false,					// WORK_STEALING: 				Equal-shares load-balancing
		9						// LOG_SCHEDULE_GRANULARITY:	512 items
	> {};



//-----------------------------------------------------------------------------
// SM1.0 default granularity parameterization type
//-----------------------------------------------------------------------------

template <>
struct TunedConfig<SM10>
	: MemcopyConfig<
		unsigned short,			// Data type					Use int16s as primary movement type
		8,						// CTA_OCCUPANCY: 				8 CTAs/SM
		7,						// LOG_THREADS: 				128 threads/CTA
		2,						// LOG_LOAD_VEC_SIZE: 			vec-4
		0,						// LOG_LOADS_PER_TILE: 			4 loads
		NONE,					// CACHE_MODIFIER: 				CA (cache all levels)
		false,					// WORK_STEALING: 				Equal-shares load-balancing
		9						// LOG_SCHEDULE_GRANULARITY:	512 items
	> {};





}// namespace large_problem_tuning
}// namespace memcopy
}// namespace b40c

