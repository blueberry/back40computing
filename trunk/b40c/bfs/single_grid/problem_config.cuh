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
#include <b40c/util/operators.cuh>

#include <b40c/bfs/compact/upsweep_kernel_config.cuh>
#include <b40c/bfs/compact/downsweep_kernel_config.cuh>
#include <b40c/bfs/expand_atomic/sweep_kernel_config.cuh>

#include <b40c/scan/downsweep_kernel_config.cuh>
#include <b40c/scan/problem_type.cuh>


namespace b40c {
namespace bfs {
namespace single_grid {




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
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	int LOG_SCHEDULE_GRANULARITY,

	// BFS expansion tunable params
	int EXPAND_LOG_LOAD_VEC_SIZE,
	int EXPAND_LOG_LOADS_PER_TILE,
	int EXPAND_LOG_RAKING_THREADS,
	util::io::ld::CacheModifier QUEUE_READ_MODIFIER,
	util::io::ld::CacheModifier COLUMN_READ_MODIFIER,
	util::io::ld::CacheModifier ROW_OFFSET_ALIGNED_READ_MODIFIER,
	util::io::ld::CacheModifier ROW_OFFSET_UNALIGNED_READ_MODIFIER,
	util::io::st::CacheModifier QUEUE_WRITE_MODIFIER,
	bool WORK_STEALING,
	int EXPAND_LOG_SCHEDULE_GRANULARITY,

	// Compaction upsweep tunable params
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,

	// Compaction spine tunable params
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Compaction downsweep tunable params
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct ProblemConfig : _ProblemType
{
	typedef _ProblemType 					ProblemType;
	typedef typename ProblemType::SizeT 	SizeT;

	//---------------------------------------------------------------------
	// BFS Expand Atomic
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction downsweep kernel
	typedef expand_atomic::SweepKernelConfig <
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
		WORK_STEALING,
		EXPAND_LOG_SCHEDULE_GRANULARITY>
			ExpandSweep;


	//---------------------------------------------------------------------
	// Upsweep
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction upsweep kernel
	typedef compact::UpsweepKernelConfig <
		ProblemType,
		CUDA_ARCH,
		MAX_CTA_OCCUPANCY,
		LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			CompactUpsweep;

	//---------------------------------------------------------------------
	// Spine
	//---------------------------------------------------------------------

	// Problem type for compaction spine
	typedef scan::ProblemType<
		SizeT,								// Spine-scan type is SizeT
		SizeT,								// Spine-scan sizet is SizeT
		true,								// Exclusive
		util::DefaultSum<SizeT>,
		util::DefaultSumIdentity<SizeT> > CompactSpineProblem;

	// Kernel config for the BFS compaction spine kernel
	typedef scan::DownsweepKernelConfig <
		CompactSpineProblem,
		CUDA_ARCH,
		1,									// Only a single-CTA grid
		LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + LOG_THREADS>
			CompactSpine;

	//---------------------------------------------------------------------
	// Downsweep
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction downsweep kernel
	typedef compact::DownsweepKernelConfig <
		ProblemType,
		CUDA_ARCH,
		MAX_CTA_OCCUPANCY,
		LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			CompactDownsweep;


	/**
	 * Shared memory structure
	 */
	union SmemStorage
	{
		typename ExpandSweep::SmemStorage 		expand_smem_storage;
		typename CompactUpsweep::SmemStorage 	upsweep_smem_storage;
		typename CompactSpine::SmemStorage 		spine_smem_storage;
		typename CompactDownsweep::SmemStorage 	downsweep_smem_storage;
	};

	enum {
		THREADS 			= 1 << LOG_THREADS,
		CTA_OCCUPANCY		= B40C_MIN(B40C_MIN(ExpandSweep::CTA_OCCUPANCY ,CompactUpsweep::CTA_OCCUPANCY), CompactDownsweep::CTA_OCCUPANCY),
	};



};
		
} // namespace single_grid
} // namespace bfs
} // namespace b40c

