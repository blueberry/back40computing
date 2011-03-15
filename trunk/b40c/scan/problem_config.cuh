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
 * Unified scan granularity configuration type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * upsweep, spine, and downsweep kernels, this type includes enactor tuning
 * parameters that define kernel-dispatch policy.  By encapsulating the tuning information
 * for dispatch and both kernels, we assure operational consistency over an entire
 * scan pass.
 */
template <
	// ProblemType type parameters
	typename ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Common tunable params
	util::ld::CacheModifier READ_MODIFIER,
	util::st::CacheModifier WRITE_MODIFIER,
	bool _UNIFORM_SMEM_ALLOCATION,
	bool _UNIFORM_GRID_SIZE,
	bool _OVERSUBSCRIBED_GRID_SIZE,
	int LOG_SCHEDULE_GRANULARITY,

	// Upsweep tunable params
	int UPSWEEP_MAX_CTA_OCCUPANCY,
	int UPSWEEP_LOG_THREADS,
	int UPSWEEP_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_LOG_LOADS_PER_TILE,
	int UPSWEEP_LOG_RAKING_THREADS,

	// Spine tunable params
	int SPINE_LOG_THREADS,
	int SPINE_LOG_LOAD_VEC_SIZE,
	int SPINE_LOG_LOADS_PER_TILE,
	int SPINE_LOG_RAKING_THREADS,

	// Downsweep tunable params
	int DOWNSWEEP_MAX_CTA_OCCUPANCY,
	int DOWNSWEEP_LOG_THREADS,
	int DOWNSWEEP_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_LOG_LOADS_PER_TILE,
	int DOWNSWEEP_LOG_RAKING_THREADS>

struct ProblemConfig
{
	// Kernel config for the upsweep reduction kernel
	typedef reduction::KernelConfig <
		ProblemType,
		CUDA_ARCH,
		UPSWEEP_MAX_CTA_OCCUPANCY,
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
	typedef KernelConfig <
		ProblemType,
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

	typedef KernelConfig <
		ProblemType,
		CUDA_ARCH,
		DOWNSWEEP_MAX_CTA_OCCUPANCY,
		DOWNSWEEP_LOG_THREADS,
		DOWNSWEEP_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_LOG_LOADS_PER_TILE,
		DOWNSWEEP_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			Downsweep;

	enum {
		UNIFORM_SMEM_ALLOCATION 	= _UNIFORM_SMEM_ALLOCATION,
		UNIFORM_GRID_SIZE 			= _UNIFORM_GRID_SIZE,
		OVERSUBSCRIBED_GRID_SIZE	= _OVERSUBSCRIBED_GRID_SIZE,
		VALID 						= Upsweep::VALID & Spine::VALID & Downsweep::VALID,
	};

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

			UPSWEEP_MAX_CTA_OCCUPANCY,
			UPSWEEP_LOG_THREADS,
			UPSWEEP_LOG_LOAD_VEC_SIZE,
			UPSWEEP_LOG_LOADS_PER_TILE,
			UPSWEEP_LOG_RAKING_THREADS,

			SPINE_LOG_THREADS,
			SPINE_LOG_LOAD_VEC_SIZE,
			SPINE_LOG_LOADS_PER_TILE,
			SPINE_LOG_RAKING_THREADS,

			DOWNSWEEP_MAX_CTA_OCCUPANCY,
			DOWNSWEEP_LOG_THREADS,
			DOWNSWEEP_LOG_LOAD_VEC_SIZE,
			DOWNSWEEP_LOG_LOADS_PER_TILE,
			DOWNSWEEP_LOG_RAKING_THREADS);
	}
};
		

}// namespace scan
}// namespace b40c

