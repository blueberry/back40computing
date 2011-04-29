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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Distribution sort upsweep kernel configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace upsweep {


/**
 * A detailed upsweep configuration type that specializes kernel code for a specific 
 * sorting pass.  It encapsulates granularity details derived from the inherited 
 * upsweep TuningConfig
 */
template <
	typename 		TuningConfig,
	typename 		PreprocessTraitsType,
	int 			_CURRENT_PASS,
	int 			_CURRENT_BIT>

struct KernelConfig : TuningConfig
{
	typedef PreprocessTraitsType					PreprocessTraits;
	
	enum {		// N.B.: We use an enum type here b/c of a NVCC-win compiler bug involving ternary expressions in static-const fields

		RADIX_DIGITS 						= 1 << TuningConfig::RADIX_BITS,
		CURRENT_PASS						= _CURRENT_PASS,
		CURRENT_BIT							= _CURRENT_BIT,

		THREADS								= 1 << TuningConfig::LOG_THREADS,

		LOG_WARPS							= TuningConfig::LOG_THREADS - B40C_LOG_WARP_THREADS(TuningConfig::CUDA_ARCH),
		WARPS								= 1 << LOG_WARPS,

		LOAD_VEC_SIZE						= 1 << TuningConfig::LOG_LOAD_VEC_SIZE,
		LOADS_PER_TILE						= 1 << TuningConfig::LOG_LOADS_PER_TILE,

		LOG_TILE_ELEMENTS_PER_THREAD		= TuningConfig::LOG_LOAD_VEC_SIZE + TuningConfig::LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD			= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 					= LOG_TILE_ELEMENTS_PER_THREAD + TuningConfig::LOG_THREADS,
		TILE_ELEMENTS						= 1 << LOG_TILE_ELEMENTS,

		// A lane is a row of 32-bit words, one words per thread, each words a
		// composite of four 8-bit digit counters, i.e., we need one lane for every
		// four radix digits.

		LOG_COMPOSITE_LANES 				= (TuningConfig::RADIX_BITS >= 2) ?
												TuningConfig::RADIX_BITS - 2 :
												0,	// Always at least one lane
		COMPOSITE_LANES 					= 1 << LOG_COMPOSITE_LANES,
	
		LOG_COMPOSITES_PER_LANE				= TuningConfig::LOG_THREADS,				// Every thread contributes one partial for each lane
		COMPOSITES_PER_LANE 				= 1 << LOG_COMPOSITES_PER_LANE,
	
		// To prevent digit-counter overflow, we must partially-aggregate the
		// 8-bit composite counters back into SizeT-bit registers periodically.  The lanes
		// are divided up amongst the warps for aggregation.  Each lane is
		// therefore equivalent to four rows of SizeT-bit digit-counts, each the width of a warp.
	
		LOG_LANES_PER_WARP					= B40C_MAX(0, LOG_COMPOSITE_LANES - LOG_WARPS),
		LANES_PER_WARP 						= 1 << LOG_LANES_PER_WARP,
	
		LOG_COMPOSITES_PER_LANE_PER_THREAD 	= LOG_COMPOSITES_PER_LANE - B40C_LOG_WARP_THREADS(TuningConfig::CUDA_ARCH),					// Number of partials per thread to aggregate
		COMPOSITES_PER_LANE_PER_THREAD 		= 1 << LOG_COMPOSITES_PER_LANE_PER_THREAD,
	
		AGGREGATED_ROWS						= RADIX_DIGITS,
		AGGREGATED_PARTIALS_PER_ROW 		= B40C_WARP_THREADS(TuningConfig::CUDA_ARCH),
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIALS_PER_ROW + 1,

		// Unroll tiles in batches of 64 elements per thread (255 is maximum without risking overflow)
		UNROLL_COUNT 						= 64 / TILE_ELEMENTS_PER_THREAD,
	};

	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			union {
				char counters[COMPOSITE_LANES][THREADS][4];
				int words[COMPOSITE_LANES][THREADS];
			} composite_counters;

			typename TuningConfig::SizeT aggregate[AGGREGATED_ROWS][PADDED_AGGREGATED_PARTIALS_PER_ROW];
		};
	};
	
	enum {
		SMEM_BYTES 							= sizeof(SmemStorage),
	};

};



} // namespace upsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

