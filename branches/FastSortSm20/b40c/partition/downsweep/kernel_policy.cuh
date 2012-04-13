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
 * Configuration policy for partitioning downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/srts_grid.cuh>

namespace b40c {
namespace partition {
namespace downsweep {


/**
 * A detailed partitioning downsweep kernel configuration policy type that specializes
 * kernel code for a specific pass.  It encapsulates tuning configuration policy
 * details derived from TuningPolicy
 */
template <typename TuningPolicy>
struct KernelPolicy : TuningPolicy
{
	typedef typename TuningPolicy::SizeT 			SizeT;
	typedef typename TuningPolicy::KeyType 			KeyType;
	typedef typename TuningPolicy::ValueType 		ValueType;
	typedef typename TuningPolicy::Counter 			Counter;
	typedef typename TuningPolicy::RakingPartial	RakingPartial;

	enum {

		LOG_BINS						= TuningPolicy::LOG_BINS,
		BINS 							= 1 << LOG_BINS,

		LOG_THREADS						= TuningPolicy::LOG_THREADS,
		THREADS							= 1 << LOG_THREADS,

		LOG_WARPS						= TuningPolicy::LOG_THREADS - B40C_LOG_WARP_THREADS(TuningPolicy::CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOG_RAKING_THREADS				= TuningPolicy::LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		LOG_RAKING_WARPS				= LOG_RAKING_THREADS - B40C_LOG_WARP_THREADS(TuningPolicy::CUDA_ARCH),
		RAKING_WARPS					= 1 << LOG_RAKING_WARPS,

		LOAD_VEC_SIZE					= 1 << TuningPolicy::LOG_LOAD_VEC_SIZE,

		LOG_TILE_ELEMENTS				= TuningPolicy::LOG_THREADS + TuningPolicy::LOG_LOAD_VEC_SIZE,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,
	
		LOG_SCAN_BINS					= TuningPolicy::LOG_BINS,
		SCAN_BINS						= 1 << LOG_SCAN_BINS,

		PACKED_COUNTERS					= sizeof(RakingPartial) / sizeof(Counter),
		LOG_PACKED_COUNTERS				= util::Log2<PACKED_COUNTERS>::VALUE,

		LOG_SCAN_LANES					= B40C_MAX((LOG_SCAN_BINS - LOG_PACKED_COUNTERS), 0),				// Always at least one lane
		SCAN_LANES						= 1 << LOG_SCAN_LANES,

		LOG_SCAN_ELEMENTS				= LOG_SCAN_LANES + LOG_THREADS,
		SCAN_ELEMENTS					= 1 << LOG_SCAN_ELEMENTS,

		LOG_BASE_RAKING_SEG				= LOG_SCAN_ELEMENTS - LOG_RAKING_THREADS,
		PADDED_RAKING_SEG				= (1 << LOG_BASE_RAKING_SEG) + 1,

		LOG_MEM_BANKS					= B40C_LOG_MEM_BANKS(TuningPolicy::CUDA_ARCH)
	};


	
	/**
	 * Shared storage for partitioning upsweep
	 */
	struct SmemStorage
	{
		SizeT							packed_offset;
		SizeT							packed_offset_limit;

		bool 							non_trivial_pass;
		util::CtaWorkLimits<SizeT> 		work_limits;

		SizeT 							bin_carry[BINS];

		// Storage for scanning local ranks
		volatile RakingPartial			warpscan[RAKING_WARPS * 2 * B40C_WARP_THREADS(CUDA_ARCH)];

		struct {
			int4						align_padding;
			union {
				Counter					packed_counters[SCAN_LANES + 1][THREADS][PACKED_COUNTERS];
				RakingPartial			raking_grid[RAKING_THREADS][PADDED_RAKING_SEG];
				KeyType 				key_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
				ValueType 				value_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
			};
		};
	};

	enum {
		THREAD_OCCUPANCY					= B40C_SM_THREADS(TuningPolicy::CUDA_ARCH) >> TuningPolicy::LOG_THREADS,
		SMEM_OCCUPANCY						= B40C_SMEM_BYTES(TuningPolicy::CUDA_ARCH) / sizeof(SmemStorage),
		MAX_CTA_OCCUPANCY					= B40C_MIN(B40C_SM_CTAS(TuningPolicy::CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),

		VALID								= (MAX_CTA_OCCUPANCY > 0),
	};


	__device__ __forceinline__ static void PreprocessKey(KeyType &key) {}

	__device__ __forceinline__ static void PostprocessKey(KeyType &key) {}
};
	


} // namespace downsweep
} // namespace partition
} // namespace b40c

