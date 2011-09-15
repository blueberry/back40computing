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
		LOADS_PER_CYCLE					= 1 << TuningPolicy::LOG_LOADS_PER_CYCLE,
		CYCLES_PER_TILE					= 1 << TuningPolicy::LOG_CYCLES_PER_TILE,

		LOG_LOADS_PER_TILE				= TuningPolicy::LOG_LOADS_PER_CYCLE +
												TuningPolicy::LOG_CYCLES_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_CYCLE_ELEMENTS				= TuningPolicy::LOG_THREADS +
												TuningPolicy::LOG_LOADS_PER_CYCLE +
												TuningPolicy::LOG_LOAD_VEC_SIZE,
		CYCLE_ELEMENTS					= 1 << LOG_CYCLE_ELEMENTS,

		LOG_TILE_ELEMENTS				= TuningPolicy::LOG_CYCLES_PER_TILE + LOG_CYCLE_ELEMENTS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_TILE_ELEMENTS - TuningPolicy::LOG_THREADS,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,
	
		LOG_SCAN_LANES_PER_CYCLE		= B40C_MAX((TuningPolicy::LOG_BINS - 2), 0),		// Always at least one lane per load
		SCAN_LANES_PER_CYCLE			= 1 << LOG_SCAN_LANES_PER_CYCLE,

		LOG_DEPOSITS_PER_LANE 			= TuningPolicy::LOG_THREADS + TuningPolicy::LOG_LOADS_PER_CYCLE,
	};


	// Smem SRTS grid type for reducing and scanning a cycle of 
	// (bins/4) lanes of composite 8-bit bin counters
	typedef util::SrtsGrid<
		TuningPolicy::CUDA_ARCH,
		int,											// Partial type
		LOG_DEPOSITS_PER_LANE,							// Deposits per lane
		LOG_SCAN_LANES_PER_CYCLE,						// Lanes (the number of composite digits)
		TuningPolicy::LOG_RAKING_THREADS,				// Raking threads
		false>											// Any prefix dependences between lanes are explicitly managed
			ByteGrid;

	
	/**
	 * Shared storage for partitioning upsweep
	 */
	struct SmemStorage
	{
		bool 							non_trivial_pass;
		util::CtaWorkLimits<SizeT> 		work_limits;

		SizeT							bin_carry[BINS];
		int								bin_inclusive[BINS];

		// Storage for scanning local ranks
		volatile int 					warpscan_low[2][B40C_WARP_THREADS(CUDA_ARCH)];
		volatile int 					warpscan_high[2][B40C_WARP_THREADS(CUDA_ARCH)];

		union {
			struct {
				int 					byte_raking_lanes[ByteGrid::RAKING_ELEMENTS];

				union {
					volatile int		exclusive_prefixes_a[2][ByteGrid::RAKING_THREADS / 2][2];
					volatile int		exclusive_prefixes_b[2][2][ByteGrid::RAKING_THREADS / 2];
					volatile int		exclusive_prefixes_c[2][ByteGrid::RAKING_THREADS];
					volatile int		exclusive_prefixes_d[4][ByteGrid::RAKING_THREADS / 2];
				};

				union {
					volatile int		inclusive_prefixes_c[2][ByteGrid::RAKING_THREADS];
					volatile int		inclusive_prefixes_d[4][ByteGrid::RAKING_THREADS / 2];
				};

			};

			KeyType 					key_exchange[TILE_ELEMENTS + 1];			// Last index is for invalid elements to be culled (if any)
			ValueType 					value_exchange[TILE_ELEMENTS + 1];
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

