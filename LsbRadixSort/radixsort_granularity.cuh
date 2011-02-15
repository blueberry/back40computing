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
 *  
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"

#include "radixsort_kernel_upsweep.cuh"
#include "radixsort_kernel_spine.cuh"
#include "radixsort_kernel_downsweep.cuh"

namespace b40c {
namespace lsb_radix_sort {

		
/**
 * Unified granularity configuration type for all three kernels in a sorting pass
 * (upsweep, spinescan, and downsweep).  
 * 
 * This C++ type encapsulates all three sets of kernel-tuning parameters (they 
 * are reflected via the static fields). By deriving from the three granularity 
 * types, we assure operational consistency over an entire sorting pass. 
 */
template <
	// Common
	typename KeyType,
	typename ValueType,
	typename IndexType,
	int RADIX_BITS,
	int LOG_SUBTILE_ELEMENTS,
	CacheModifier CACHE_MODIFIER,
	bool EARLY_EXIT,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	
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

struct GranularityConfig
{
	// Unsigned integer type to cast keys as in order to make them suitable 
	// for radix sorting 
	typedef typename KeyTraits<KeyType>::ConvertedKeyType ConvertedKeyType;

	static const bool UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION;
	static const bool UNIFORM_GRID_SIZE 		= _UNIFORM_GRID_SIZE;
	
	typedef upsweep::UpsweepConfig<
		ConvertedKeyType, 
		IndexType,
		RADIX_BITS, 
		LOG_SUBTILE_ELEMENTS,
		UPSWEEP_CTA_OCCUPANCY,  
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,  	
		UPSWEEP_LOG_LOADS_PER_TILE,
		CACHE_MODIFIER,
		EARLY_EXIT>
			Upsweep;
	
	typedef spine_scan::SpineScanConfig<
		IndexType,								// Type of scan problem
		int,									// Type for indexing into scan problem
		SPINE_CTA_OCCUPANCY,
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		CACHE_MODIFIER>
			SpineScan;
	
	typedef downsweep::DownsweepConfig<
		ConvertedKeyType,
		ValueType,
		IndexType,
		RADIX_BITS,
		LOG_SUBTILE_ELEMENTS,
		DOWNSWEEP_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_CYCLE,
		DOWNSWEEP_LOG_CYCLES_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		CACHE_MODIFIER,
		EARLY_EXIT>
			Downsweep;
};
		

}// namespace lsb_radix_sort
}// namespace b40c

