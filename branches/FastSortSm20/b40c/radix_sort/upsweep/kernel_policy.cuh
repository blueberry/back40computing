/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * Configuration policy for partitioning upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/kernel_policy.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {


/**
 * Radix sort upsweep reduction tuning policy.
 */
template <
	int 							_LOG_SCHEDULE_GRANULARITY,
	typename 						_ProblemType,
	int 							_LOG_BINS,
	int 							_MIN_CTA_OCCUPANCY,
	int 							_LOG_THREADS,
	int 							_LOG_LOAD_VEC_SIZE,
	int 							_LOG_LOADS_PER_TILE,
	util::io::ld::CacheModifier 	_READ_MODIFIER,
	util::io::st::CacheModifier 	_WRITE_MODIFIER,
	bool 							_EARLY_EXIT>
struct KernelPolicy : _ProblemType
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	typedef typename ProblemType::SizeT SizeT;

	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------


	enum {
		MIN_CTA_OCCUPANCY  					= _MIN_CTA_OCCUPANCY,

		LOG_SCHEDULE_GRANULARITY			= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY				= 1 << LOG_SCHEDULE_GRANULARITY,

		LOG_BINS							= _LOG_BINS,
		BINS 								= 1 << LOG_BINS,

		LOG_THREADS 						= _LOG_THREADS,
		THREADS								= 1 << LOG_THREADS,

		LOG_LOAD_VEC_SIZE  					= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE						= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 					= _LOG_LOADS_PER_TILE,
		LOADS_PER_TILE						= 1 << LOG_LOADS_PER_TILE,

		LOG_WARPS							= LOG_THREADS - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__),
		WARPS								= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD		= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD			= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 					= LOG_TILE_ELEMENTS_PER_THREAD + LOG_THREADS,
		TILE_ELEMENTS						= 1 << LOG_TILE_ELEMENTS,


		// A shared-memory composite counter lane is a row of 32-bit words, one word per thread, each word a
		// composite of four 8-bit bin counters.  I.e., we need one lane for every four distribution bins.

		LOG_COMPOSITE_LANES 				= (LOG_BINS >= 2) ?
												LOG_BINS - 2 :
												0,	// Always at least one lane
		COMPOSITE_LANES 					= 1 << LOG_COMPOSITE_LANES,

		LOG_COMPOSITES_PER_LANE				= LOG_THREADS,				// Every thread contributes one partial for each lane
		COMPOSITES_PER_LANE 				= 1 << LOG_COMPOSITES_PER_LANE,

		// To prevent bin-counter overflow, we must partially-aggregate the
		// 8-bit composite counters back into SizeT-bit registers periodically.  Each lane
		// is assigned to a warp for aggregation.  Each lane is therefore equivalent to
		// four rows of SizeT-bit bin-counts, each the width of a warp.

		LOG_LANES_PER_WARP					= CUB_MAX(0, LOG_COMPOSITE_LANES - LOG_WARPS),
		LANES_PER_WARP 						= 1 << LOG_LANES_PER_WARP,

		LOG_COMPOSITES_PER_LANE_PER_THREAD 	= LOG_COMPOSITES_PER_LANE - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__),					// Number of partials per thread to aggregate
		COMPOSITES_PER_LANE_PER_THREAD 		= 1 << LOG_COMPOSITES_PER_LANE_PER_THREAD,

		AGGREGATED_ROWS						= BINS,
		AGGREGATED_PARTIALS_PER_ROW 		= B40C_WARP_THREADS(__B40C_CUDA_ARCH__),
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIALS_PER_ROW + 1,

		// Unroll tiles in batches of X elements per thread (X = log(255) is maximum without risking overflow)
		LOG_UNROLL_COUNT 					= 6 - LOG_TILE_ELEMENTS_PER_THREAD,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;

	//---------------------------------------------------------------------
	// CTA storage type definition
	//---------------------------------------------------------------------

	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			// Composite counter storage
			union {
				char counters[COMPOSITE_LANES][THREADS][4];
				int words[COMPOSITE_LANES][THREADS];
				int direct[COMPOSITE_LANES * THREADS];
			} composite_counters;

			// Final bin reduction storage
			typename TuningPolicy::SizeT aggregate[AGGREGATED_ROWS][PADDED_AGGREGATED_PARTIALS_PER_ROW];
		};
	};

};
	


} // namespace upsweep
} // namespace radix_sort
} // namespace b40c

