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
 * Configuration policy for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/partition/downsweep/kernel_policy.cuh>

namespace b40c {
namespace radix_sort {
namespace downsweep {

/**
 * Types of scattering strategies
 */
enum ScatterStrategy {
	SCATTER_DIRECT = 0,
	SCATTER_TWO_PHASE,
	SCATTER_WARP_TWO_PHASE,
};


/**
 * Downsweep tuning policy.
 */
template <
	int 							_LOG_SCHEDULE_GRANULARITY,
	typename 						_ProblemType,
	int 							_LOG_BINS,
	int 							_MIN_CTA_OCCUPANCY,
	int 							_LOG_THREADS,
	int 							_LOG_LOAD_VEC_SIZE,
	util::io::ld::CacheModifier	 	_READ_MODIFIER,
	util::io::st::CacheModifier 	_WRITE_MODIFIER,
	ScatterStrategy 				_SCATTER_STRATEGY,
	bool						 	_EARLY_EXIT>
struct KernelPolicy : _ProblemType
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	typedef typename ProblemType::SizeT 			SizeT;
	typedef typename ProblemType::KeyType 			KeyType;
	typedef typename ProblemType::ValueType 		ValueType;
	typedef unsigned short							Counter;			// Integer type for digit counters (to be packed in the RakingPartial type defined below)
	typedef unsigned int							RakingPartial;		// Integer type for raking partials (packed counters).  Consider using 64b counters on Kepler+ for better smem bandwidth


	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {

		MIN_CTA_OCCUPANCY  				= _MIN_CTA_OCCUPANCY,

		LOG_SCHEDULE_GRANULARITY		= _LOG_SCHEDULE_GRANULARITY,
		SCHEDULE_GRANULARITY			= 1 << LOG_SCHEDULE_GRANULARITY,

		LOG_BINS						= _LOG_BINS,
		BINS 							= 1 << LOG_BINS,

		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_WARP_THREADS 				= B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__),
		WARP_THREADS					= 1 << LOG_WARP_THREADS,

		LOG_WARPS						= LOG_THREADS - LOG_WARP_THREADS,
		WARPS							= 1 << LOG_WARPS,

		LOG_LOAD_VEC_SIZE 				= _LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_TILE_ELEMENTS				= LOG_THREADS + LOG_LOAD_VEC_SIZE,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		PACKED_COUNTERS					= sizeof(RakingPartial) / sizeof(Counter),
		LOG_PACKED_COUNTERS				= util::Log2<PACKED_COUNTERS>::VALUE,

		LOG_SCAN_LANES					= CUB_MAX((LOG_BINS - LOG_PACKED_COUNTERS), 0),				// Always at least one lane
		SCAN_LANES						= 1 << LOG_SCAN_LANES,

		LOG_SCAN_ELEMENTS				= LOG_SCAN_LANES + LOG_THREADS,
		SCAN_ELEMENTS					= 1 << LOG_SCAN_ELEMENTS,

		LOG_BASE_RAKING_SEG				= LOG_SCAN_ELEMENTS - LOG_THREADS,
		PADDED_RAKING_SEG				= (1 << LOG_BASE_RAKING_SEG) + 1,

		LOG_MEM_BANKS					= B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__),

		LOG_PACK_SIZE 					= 2,
		PACK_SIZE						= 1 << LOG_PACK_SIZE,

		BANK_PADDING 					= 1,		// Whether or not to insert padding for exchanging keys
	};

	static const util::io::ld::CacheModifier READ_MODIFIER 		= _READ_MODIFIER;
	static const util::io::st::CacheModifier WRITE_MODIFIER 	= _WRITE_MODIFIER;
	static const ScatterStrategy SCATTER_STRATEGY 				= _SCATTER_STRATEGY;


	//---------------------------------------------------------------------
	// CTA storage type definition
	//---------------------------------------------------------------------

	/**
	 * Shared storage for partitioning downsweep
	 */
	struct SmemStorage
	{
		SizeT							packed_offset;
		SizeT							packed_offset_limit;

		bool 							non_trivial_pass;
		util::CtaWorkLimits<SizeT> 		work_limits;

		SizeT 							bin_carry[BINS];

		// Storage for scanning local ranks
		volatile RakingPartial			warpscan[WARPS][WARP_THREADS * 3 / 2];

		struct {
			int4						align_padding;
			union {
				Counter					packed_counters[SCAN_LANES + 1][THREADS][PACKED_COUNTERS];
				RakingPartial			raking_grid[THREADS][PADDED_RAKING_SEG];
				KeyType 				key_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
				ValueType 				value_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
			};
		};
	};

};



} // namespace downsweep
} // namespace partition
} // namespace b40c

