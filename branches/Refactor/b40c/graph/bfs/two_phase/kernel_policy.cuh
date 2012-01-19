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
 * "Metatype" for guiding BFS contract-expand granularity configuration
 ******************************************************************************/

#pragma once

#include <b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace two_phase {

/**
 *
 */
template <
	// ProblemType type parameters
	typename _ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Behavioral control parameters
	bool _INSTRUMENT,					// Whether or not we want instrumentation logic generated
	int _SATURATION_QUIT,				// If positive, signal that we're done with two-phase iterations if problem size drops below (SATURATION_QUIT * grid_size * TILE_SIZE)

	// Tunable parameters
	int _MAX_CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_TILE,
	int _LOG_RAKING_THREADS,
	util::io::ld::CacheModifier _QUEUE_READ_MODIFIER,
	util::io::ld::CacheModifier _COLUMN_READ_MODIFIER,
	util::io::ld::CacheModifier _ROW_OFFSET_ALIGNED_READ_MODIFIER,
	util::io::ld::CacheModifier _ROW_OFFSET_UNALIGNED_READ_MODIFIER,
	util::io::st::CacheModifier _QUEUE_WRITE_MODIFIER,
	bool _WORK_STEALING,
	int _WARP_GATHER_THRESHOLD,
	int _CTA_GATHER_THRESHOLD,
	int _LOG_SCHEDULE_GRANULARITY>

struct KernelPolicy : _ProblemType
{
	typedef _ProblemType 							ProblemType;

	typedef typename ProblemType::VertexId 			VertexId;
	typedef typename ProblemType::SizeT 			SizeT;
	typedef typename ProblemType::VisitedMask 	VisitedMask;

	typedef contract_atomic::KernelPolicy<
		_ProblemType,
		CUDA_ARCH,
		_INSTRUMENT,
		true,
		_MAX_CTA_OCCUPANCY,
		_LOG_THREADS,
		_LOG_LOAD_VEC_SIZE,
		2, //_LOG_LOADS_PER_TILE,
		_LOG_RAKING_THREADS,
		_QUEUE_READ_MODIFIER,
		_QUEUE_WRITE_MODIFIER,
		_WORK_STEALING,
		_LOG_SCHEDULE_GRANULARITY> CompactKernelPolicy;

	typedef expand_atomic::KernelPolicy<
		_ProblemType,
		CUDA_ARCH,
		_INSTRUMENT,
		_SATURATION_QUIT,
		_MAX_CTA_OCCUPANCY,
		_LOG_THREADS,
		_LOG_LOAD_VEC_SIZE,
		_LOG_LOADS_PER_TILE,
		_LOG_RAKING_THREADS,
		_QUEUE_READ_MODIFIER,
		_COLUMN_READ_MODIFIER,
		_ROW_OFFSET_ALIGNED_READ_MODIFIER,
		_ROW_OFFSET_UNALIGNED_READ_MODIFIER,
		_QUEUE_WRITE_MODIFIER,
		_WORK_STEALING,
		_WARP_GATHER_THRESHOLD,
		_CTA_GATHER_THRESHOLD,
		_LOG_SCHEDULE_GRANULARITY> ExpandKernelPolicy;

	enum {
		INSTRUMENT						= _INSTRUMENT,
		LOG_THREADS 					= _LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,
	};

	/**
	 * Shared memory structure
	 */
	union SmemStorage
	{
		typename CompactKernelPolicy::SmemStorage 	contract;
		typename ExpandKernelPolicy::SmemStorage 	expand;
	};

	enum {
		// Total number of smem quads needed by this kernel
		SMEM_QUADS						= B40C_QUADS(sizeof(SmemStorage)),

		THREAD_OCCUPANCY				= B40C_SM_THREADS(CUDA_ARCH) >> _LOG_THREADS,
		SMEM_OCCUPANCY					= B40C_SMEM_BYTES(CUDA_ARCH) / (SMEM_QUADS * sizeof(uint4)),
		CTA_OCCUPANCY  					= B40C_MIN(_MAX_CTA_OCCUPANCY, B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY))),

		VALID							= (CTA_OCCUPANCY > 0),
	};
};


} // namespace two_phase
} // namespace bfs
} // namespace graph
} // namespace b40c

