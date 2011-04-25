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
 * Distribution sort downsweep kernel configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/srts_grid.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace downsweep {


/**
 * A detailed downsweep configuration type that specializes kernel code for a 
 * specific sorting pass.  It encapsulates granularity details derived from the 
 * inherited downsweep TuningConfig
 */
template <
	typename 		TuningConfig,
	typename 		PreprocessTraitsType, 
	typename 		PostprocessTraitsType, 
	int 			_CURRENT_PASS,
	int 			_CURRENT_BIT>
struct KernelConfig : TuningConfig
{
	typedef PreprocessTraitsType					PreprocessTraits;
	typedef PostprocessTraitsType					PostprocessTraits;

	enum {

		RADIX_DIGITS 					= 1 << TuningConfig::RADIX_BITS,
		CURRENT_PASS					= _CURRENT_PASS,
		CURRENT_BIT						= _CURRENT_BIT,

		THREADS							= 1 << TuningConfig::LOG_THREADS,

		LOG_WARPS						= TuningConfig::LOG_THREADS - B40C_LOG_WARP_THREADS(TuningConfig::CUDA_ARCH),
		WARPS							= 1 << LOG_WARPS,

		LOAD_VEC_SIZE					= 1 << TuningConfig::LOG_LOAD_VEC_SIZE,
		LOADS_PER_CYCLE					= 1 << TuningConfig::LOG_LOADS_PER_CYCLE,
		CYCLES_PER_TILE					= 1 << TuningConfig::LOG_CYCLES_PER_TILE,

		LOG_LOADS_PER_TILE				= TuningConfig::LOG_LOADS_PER_CYCLE +
												TuningConfig::LOG_CYCLES_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_CYCLE_ELEMENTS				= TuningConfig::LOG_THREADS +
												TuningConfig::LOG_LOADS_PER_CYCLE +
												TuningConfig::LOG_LOAD_VEC_SIZE,
		CYCLE_ELEMENTS					= 1 << LOG_CYCLE_ELEMENTS,

		LOG_TILE_ELEMENTS				= TuningConfig::LOG_CYCLES_PER_TILE + LOG_CYCLE_ELEMENTS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_TILE_ELEMENTS - TuningConfig::LOG_THREADS,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,
	
		LOG_SCAN_LANES_PER_LOAD			= B40C_MAX((TuningConfig::RADIX_BITS - 2), 0),		// Always at least one lane per load
		SCAN_LANES_PER_LOAD				= 1 << LOG_SCAN_LANES_PER_LOAD,

		LOG_SCAN_LANES_PER_CYCLE		= TuningConfig::LOG_LOADS_PER_CYCLE + LOG_SCAN_LANES_PER_LOAD,
		SCAN_LANES_PER_CYCLE			= 1 << LOG_SCAN_LANES_PER_CYCLE,
	};


	// Smem SRTS grid type for reducing and scanning a cycle of 
	// (radix-digits/4) lanes of composite 8-bit digit counters
	typedef util::SrtsGrid<
		TuningConfig::CUDA_ARCH,
		int,									// Partial type
		TuningConfig::LOG_THREADS,				// Depositing threads (the CTA size)
		LOG_SCAN_LANES_PER_CYCLE,				// Lanes (the number of loads)
		TuningConfig::LOG_RAKING_THREADS,		// Raking threads
		false>									// Any prefix dependences between lanes are explicitly managed
			Grid;

	
	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		typedef typename TuningConfig::SizeT 		SizeT;
		typedef typename TuningConfig::KeyType 		KeyType;
		typedef typename TuningConfig::ValueType 	ValueType;

		int 							lanes_warpscan[SCAN_LANES_PER_CYCLE][3][Grid::RAKING_THREADS_PER_LANE];		// One warpscan per lane
		SizeT							digit_carry[RADIX_DIGITS];
		int 							digit_warpscan[2][RADIX_DIGITS];
		int 							digit_prefixes[CYCLES_PER_TILE][LOADS_PER_CYCLE][RADIX_DIGITS];
		int 							lane_totals[CYCLES_PER_TILE][SCAN_LANES_PER_CYCLE];
		bool 							non_trivial_digit_pass;
		int 							selector;
		util::CtaWorkLimits<SizeT> 		work_limits;

		union {
			int 						raking_lanes[Grid::RAKING_ELEMENTS];
			KeyType 					key_exchange[TILE_ELEMENTS];
			ValueType 					value_exchange[TILE_ELEMENTS];
		} smem_pool;
	};

	enum {
		SMEM_BYTES									= sizeof(SmemStorage),
	};
};
	


} // namespace downsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

