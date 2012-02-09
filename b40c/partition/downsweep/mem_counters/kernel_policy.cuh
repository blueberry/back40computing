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
	typedef typename TuningPolicy::SizeT 		SizeT;
	typedef typename TuningPolicy::KeyType 		KeyType;
	typedef typename TuningPolicy::ValueType 	ValueType;

	enum {

		BINS 							= 1 << TuningPolicy::LOG_BINS,
		THREADS							= 1 << TuningPolicy::LOG_THREADS,

		LOG_WARPS						= TuningPolicy::LOG_THREADS - B40C_LOG_WARP_THREADS(TuningPolicy::CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOAD_VEC_SIZE					= 1 << TuningPolicy::LOG_LOAD_VEC_SIZE,

		LOG_TILE_ELEMENTS				= TuningPolicy::LOG_THREADS + TuningPolicy::LOG_LOAD_VEC_SIZE,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,
	
		LOG_SCAN_BINS					= TuningPolicy::LOG_BINS,
		SCAN_BINS						= 1 << LOG_SCAN_BINS,

		LOG_SCAN_LANES					= B40C_MAX((LOG_SCAN_BINS - 1), 0),				// Always at least one lane
		SCAN_LANES						= 1 << LOG_SCAN_LANES,

		LOG_RAKING_LANES				= B40C_MAX((LOG_SCAN_LANES - 1), 0),	// Always at least one lane
		RAKING_LANES					= 1 << LOG_RAKING_LANES,

		LOG_DEPOSITS_PER_LANE 			= TuningPolicy::LOG_THREADS,
	};


	// Smem SRTS grid type for reducing and scanning a tile of
	// (bins/2) lanes of composite 16-bit bin counters
	typedef util::SrtsGrid<
		TuningPolicy::CUDA_ARCH,
		int,											// Partial type
		LOG_DEPOSITS_PER_LANE,							// Deposits per lane
		LOG_RAKING_LANES,						// Lanes (the number of composite digits)
		TuningPolicy::LOG_RAKING_THREADS,				// Raking threads
		false>											// Any prefix dependences between lanes are explicitly managed
			RakingGrid;

	
	/**
	 * Shared storage for partitioning upsweep
	 */
	struct SmemStorage
	{
		SizeT							packed_offset;
		SizeT							packed_offset_limit;

		bool 							non_trivial_pass;
		util::CtaWorkLimits<SizeT> 		work_limits;

		SizeT							bin_carry[BINS];
		volatile int 					bin_prefixes[BINS + 1];

		// Storage for scanning local ranks
		volatile int 					warpscan[2][B40C_WARP_THREADS(CUDA_ARCH) * 3 / 2];

		union {
			struct {
				union {
					int 				packed_counters_32[SCAN_LANES][THREADS];
					short 				packed_counters_16[SCAN_LANES][THREADS][2];

					int2 				paired_counters_64[RAKING_LANES][THREADS];
					int 				paired_counters_32[RAKING_LANES][THREADS][2];
				};
				int 					raking_lanes[RakingGrid::RAKING_ELEMENTS];
			};

			KeyType 					key_exchange[TILE_ELEMENTS + (TILE_ELEMENTS / 32)];			// Last index is for invalid elements to be culled (if any)
		};
	};

	enum {
		THREAD_OCCUPANCY					= B40C_SM_THREADS(CUDA_ARCH) >> TuningPolicy::LOG_THREADS,
		SMEM_OCCUPANCY						= B40C_SMEM_BYTES(CUDA_ARCH) / sizeof(SmemStorage),
		MAX_CTA_OCCUPANCY					= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), B40C_MIN(THREAD_OCCUPANCY, SMEM_OCCUPANCY)),

		VALID								= (MAX_CTA_OCCUPANCY > 0),
	};


	__device__ __forceinline__ static void PreprocessKey(KeyType &key) {}

	__device__ __forceinline__ static void PostprocessKey(KeyType &key) {}
};
	


} // namespace downsweep
} // namespace partition
} // namespace b40c

