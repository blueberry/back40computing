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
 *  BFS Compaction Granularity Configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/bfs/compact_atomic/kernel_config.cuh>
#include <b40c/bfs/expand_atomic/kernel_config.cuh>

namespace b40c {
namespace bfs {
namespace hybrid {


/**
 * Unified BFS single-grid granularity configuration type.
 */
template <
	// ProblemType type parameters
	typename _ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Common tunable params
	int MAX_CTA_OCCUPANCY,
	int LOG_THREADS,

	// BFS expansion tunable params
	int EXPAND_LOG_LOAD_VEC_SIZE,
	int EXPAND_LOG_LOADS_PER_TILE,
	int EXPAND_LOG_RAKING_THREADS,
	util::io::ld::CacheModifier COLUMN_READ_MODIFIER,
	util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER,
	util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER,
	bool EXPAND_WORK_STEALING,
	int EXPAND_LOG_SCHEDULE_GRANULARITY,

	// Compaction upsweep tunable params
	int COMPACT_LOG_LOAD_VEC_SIZE,
	int COMPACT_LOG_LOADS_PER_TILE,
	int COMPACT_LOG_RAKING_THREADS,
	bool COMPACT_WORK_STEALING,
	int COMPACT_LOG_SCHEDULE_GRANULARITY>

struct KernelConfig : _ProblemType
{
	typedef _ProblemType 					ProblemType;
	typedef typename ProblemType::SizeT 	SizeT;

	static const util::io::ld::CacheModifier QUEUE_READ_MODIFIER = util::io::ld::cg;
	static const util::io::st::CacheModifier QUEUE_WRITE_MODIFIER = util::io::st::cg;

	//---------------------------------------------------------------------
	// Expand
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction downsweep kernel
	typedef expand_atomic::KernelConfig <
		ProblemType,
		CUDA_ARCH,
		MAX_CTA_OCCUPANCY,
		LOG_THREADS,
		EXPAND_LOG_LOAD_VEC_SIZE,
		EXPAND_LOG_LOADS_PER_TILE,
		EXPAND_LOG_RAKING_THREADS,
		QUEUE_READ_MODIFIER,
		COLUMN_READ_MODIFIER,
		ROW_OFFSET_ALIGNED_READ_MODIFIER,
		ROW_OFFSET_UNALIGNED_READ_MODIFIER,
		QUEUE_WRITE_MODIFIER,
		EXPAND_WORK_STEALING,
		EXPAND_LOG_SCHEDULE_GRANULARITY>
			ExpandConfig;


	//---------------------------------------------------------------------
	// Compact
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction upsweep kernel
	typedef compact_atomic::KernelConfig <
		ProblemType,
		CUDA_ARCH,
		MAX_CTA_OCCUPANCY,
		LOG_THREADS,
		COMPACT_LOG_LOAD_VEC_SIZE,
		COMPACT_LOG_LOADS_PER_TILE,
		COMPACT_LOG_RAKING_THREADS,
		QUEUE_READ_MODIFIER,
		QUEUE_WRITE_MODIFIER,
		COMPACT_WORK_STEALING,
		COMPACT_LOG_SCHEDULE_GRANULARITY>
			CompactConfig;


	/**
	 * Shared memory structure
	 */
	union SmemStorage
	{
		typename ExpandConfig::SmemStorage 		expand;
		typename CompactConfig::SmemStorage 	compact;
	};

	enum {
		LOG_THREADS 		= LOG_THREADS,
		THREADS 			= 1 << LOG_THREADS,
		CTA_OCCUPANCY		= B40C_MIN(ExpandConfig::CTA_OCCUPANCY ,CompactConfig::CTA_OCCUPANCY),
	};



};
		
} // namespace hybrid
} // namespace bfs
} // namespace b40c

