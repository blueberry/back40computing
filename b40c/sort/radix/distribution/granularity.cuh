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
 *  Scan Granularity Configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>
#include <b40c/reduction/kernel_config.cuh>
#include <b40c/scan/kernel_config.cuh>

namespace b40c {
namespace scan {


/**
 * Unified sort granularity configuration type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * upsweep, spine, and downsweep kernels, this type includes enactor tuning
 * parameters that define kernel-dispatch policy.  By encapsulating the tuning information
 * for dispatch and both kernels, we assure operational consistency over an entire
 * scan pass.
 */
template <
	typename ScanProblemType,

	// Common
	util::ld::CacheModifier READ_MODIFIER,
	util::st::CacheModifier WRITE_MODIFIER,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,
	int LOG_SCHEDULE_GRANULARITY,

	// Upsweep
	int UPSWEEP_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,
	int UPSWEEP_LOG_RAKING_THREADS,

	// Spine
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Downsweep
	int DOWNSWEEP_CTA_OCCUPANCY,
	int DOWNSWEEP_LOG_THREADS,
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct ScanConfig
{
	static const bool UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION;
	static const bool UNIFORM_GRID_SIZE 		= _UNIFORM_GRID_SIZE;
	static const bool OVERSUBSCRIBED_GRID_SIZE	= _OVERSUBSCRIBED_GRID_SIZE;

	// Kernel config for the upsweep reduction kernel
	typedef reduction::ReductionKernelConfig <
		ScanProblemType,
		UPSWEEP_CTA_OCCUPANCY,
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,
		UPSWEEP_LOG_LOADS_PER_TILE,
		UPSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		false,								// No workstealing: upsweep and downsweep CTAs need to process the same tiles
		LOG_SCHEDULE_GRANULARITY>
			Upsweep;

	// Kernel config for the spine scan kernel
	typedef ScanKernelConfig <
		ScanProblemType,
		1,									// Only a single-CTA grid
		SPINE_LOG_THREADS,
		SPINE_LOG_LOAD_VEC_SIZE,
		SPINE_LOG_LOADS_PER_TILE,
		SPINE_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_LOG_LOADS_PER_TILE + SPINE_LOG_LOAD_VEC_SIZE + SPINE_LOG_THREADS>
			Spine;

	typedef ScanKernelConfig <
		ScanProblemType,
		DOWNSWEEP_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			Downsweep;

	static void Print()
	{
		printf("%s, %s, %s, %s, %s, %d, "
				"%d, %d, %d, %d, %d, "
				"%d, %d, %d, %d, "
				"%d, %d, %d, %d, %d",

			CacheModifierToString((int) READ_MODIFIER),
			CacheModifierToString((int) WRITE_MODIFIER),
			(UNIFORM_SMEM_ALLOCATION) ? "true" : "false",
			(UNIFORM_GRID_SIZE) ? "true" : "false",
			(OVERSUBSCRIBED_GRID_SIZE) ? "true" : "false",
			LOG_SCHEDULE_GRANULARITY,

			UPSWEEP_CTA_OCCUPANCY,
			UPSWEEP_LOG_THREADS,
			UPSWEEP_LOG_LOAD_VEC_SIZE,
			UPSWEEP_LOG_LOADS_PER_TILE,
			UPSWEEP_LOG_RAKING_THREADS,

			SPINE_LOG_THREADS,
			SPINE_LOG_LOAD_VEC_SIZE,
			SPINE_LOG_LOADS_PER_TILE,
			SPINE_LOG_RAKING_THREADS,

			DOWNSWEEP_CTA_OCCUPANCY,
			DOWNSWEEP_LOG_THREADS,
			DOWNSWEEP_LOG_LOAD_VEC_SIZE,
			DOWNSWEEP_LOG_LOADS_PER_TILE,
			DOWNSWEEP_LOG_RAKING_THREADS);
	}
};










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

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"

#include "radixsort_kernel_upsweep.cuh"
#include "radixsort_kernel_spine.cuh"
#include "radixsort_kernel_downsweep.cuh"

namespace b40c {
namespace radix_sort {


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
	// Common
	typename KeyType,
	typename ValueType,
	typename SizeT,
	int RADIX_BITS,
	int LOG_SCHEDULE_GRANULARITY,
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

struct LsbSortConfig
{
	// Unsigned integer type to cast keys as in order to make them suitable 
	// for radix sorting 
	typedef typename KeyTraits<KeyType>::ConvertedKeyType ConvertedKeyType;

	static const bool UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION;
	static const bool UNIFORM_GRID_SIZE 		= _UNIFORM_GRID_SIZE;
	
	typedef upsweep::UpsweepConfig<
		ConvertedKeyType, 
		SizeT,
		RADIX_BITS, 
		LOG_SCHEDULE_GRANULARITY,
		UPSWEEP_CTA_OCCUPANCY,  
		UPSWEEP_LOG_THREADS,
		UPSWEEP_LOG_LOAD_VEC_SIZE,  	
		UPSWEEP_LOG_LOADS_PER_TILE,
		CACHE_MODIFIER,
		EARLY_EXIT>
			Upsweep;
	
	typedef spine_scan::SpineScanConfig<
		SizeT,								// Type of scan problem
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
		SizeT,
		RADIX_BITS,
		LOG_SCHEDULE_GRANULARITY,
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
		

}// namespace radix_sort
}// namespace b40c

