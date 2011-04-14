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
 * Copy Problem Granularity Configuration Meta-type
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/copy/sweep_kernel_config.cuh>

namespace b40c {
namespace copy {


/**
 * Unified copy problem granularity configuration type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * sweep and spine kernels, this type includes enactor tuning parameters that
 * define kernel-dispatch policy.  By encapsulating the tuning information
 * for dispatch and both kernels, we assure operational consistency over an entire
 * copy pass.
 */
template <
	// ProblemType type parameters
	typename T,
	typename SizeT,

	// Machine parameters
	int CUDA_ARCH,

	// Common tunable params
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool WORK_STEALING,
	bool _OVERSUBSCRIBED_GRID_SIZE,

	// Upsweep tunable params
	int MAX_CTA_OCCUPANCY,
	int LOG_THREADS,
	int LOG_LOAD_VEC_SIZE,
	int LOG_LOADS_PER_TILE,
	int LOG_SCHEDULE_GRANULARITY>

struct ProblemConfig
{
	static const bool OVERSUBSCRIBED_GRID_SIZE = _OVERSUBSCRIBED_GRID_SIZE;

	typedef ProblemType<T, SizeT> Problem;

	// Kernel config for the sweep copy kernel
	typedef SweepKernelConfig <
		Problem,
		CUDA_ARCH,
		MAX_CTA_OCCUPANCY,
		LOG_THREADS,
		LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		WORK_STEALING,
		LOG_SCHEDULE_GRANULARITY>
			Sweep;

	static const int VALID = Sweep::VALID;

	static void Print()
	{
		printf("%s, %s, %s, %s, %d, %d, %d, %d, %d",
			CacheModifierToString((int) READ_MODIFIER),
			CacheModifierToString((int) WRITE_MODIFIER),
			(WORK_STEALING) ? "true" : "false",
			(OVERSUBSCRIBED_GRID_SIZE) ? "true" : "false",
			MAX_CTA_OCCUPANCY,
			LOG_THREADS,
			LOG_LOAD_VEC_SIZE,
			LOG_LOADS_PER_TILE,
			LOG_SCHEDULE_GRANULARITY);
	}
};
		

}// namespace copy
}// namespace b40c

