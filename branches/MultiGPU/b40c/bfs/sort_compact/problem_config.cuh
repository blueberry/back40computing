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
 * LSB Sorting Granularity Configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/operators.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>

#include <b40c/radix_sort/distribution/upsweep/tuning_config.cuh>
#include <b40c/radix_sort/distribution/downsweep/tuning_config.cuh>
#include <b40c/radix_sort/sort_utils.cuh>

#include <b40c/scan/problem_type.cuh>
#include <b40c/scan/downsweep_kernel_config.cuh>

namespace b40c {
namespace bfs {
namespace sort_compact {


/******************************************************************************
 * Meta-type for Sorting Granularity Configuration
 ******************************************************************************/

/**
 * Unified granularity configuration type for all three kernels in a sorting pass
 * (upsweep, spinescan, and downsweep).  
 * 
 * This C++ type encapsulates all three sets of kernel-tuning parameters (they 
 * are reflected via the static fields). By deriving from the three granularity 
 * types, we assure operational consistency over an entire sorting pass. 
 */
template <
	// Problem Type
	typename KeyType,
	typename ValueType,
	typename SizeT,

	// Common
	int CUDA_ARCH,
	int RADIX_BITS,
	int LOG_SCHEDULE_GRANULARITY,
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool EARLY_EXIT,
	
	// Upsweep
	int UPSWEEP_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,
	
	// Spine-scan
	int SPINE_CTA_OCCUPANCY,
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Downsweep
	int DOWNSWEEP_CTA_OCCUPANCY,
	int DOWNSWEEP_LOG_THREADS,
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_CYCLE,
	int DOWNSWEEP_LOG_CYCLES_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct ProblemConfig
{
	typedef radix_sort::distribution::upsweep::TuningConfig<
		KeyType,
		SizeT,
		CUDA_ARCH,
		RADIX_BITS, 
		LOG_SCHEDULE_GRANULARITY,
		UPSWEEP_CTA_OCCUPANCY,  
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,  	
		UPSWEEP_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		EARLY_EXIT>
			Upsweep;

	// Problem type for spine scan
	typedef scan::ProblemType<
		SizeT,
		int,
		true,								// Exclusive
		util::DefaultSum<SizeT>,
		util::DefaultSumIdentity<SizeT> > SpineProblemType;

	// Kernel config for spine scan
	typedef scan::DownsweepKernelConfig <
		SpineProblemType,
		CUDA_ARCH,
		1,									// Only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Spine;
	
	typedef radix_sort::distribution::downsweep::TuningConfig<
		KeyType,
		ValueType,
		SizeT,
		CUDA_ARCH,
		RADIX_BITS,
		LOG_SCHEDULE_GRANULARITY,
		DOWNSWEEP_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_CYCLE,
		DOWNSWEEP_LOG_CYCLES_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		EARLY_EXIT>
			Downsweep;
};
		
} // namespace sort_compact
} // namespace bfs
} // namespace b40c

