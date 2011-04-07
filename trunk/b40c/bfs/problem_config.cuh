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
 *  BFS Compaction Granularity Configuration
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/operators.cuh>

#include <b40c/bfs/compact/upsweep_kernel_config.cuh>
#include <b40c/bfs/compact/downsweep_kernel_config.cuh>
#include <b40c/bfs/expand_atomic/sweep_kernel_config.cuh>

#include <b40c/scan/downsweep_kernel_config.cuh>
#include <b40c/scan/problem_type.cuh>


namespace b40c {
namespace bfs {


/**
 * Unified BFS granularity configuration type.
 *
 * In addition to kernel tuning parameters that guide the kernel compilation for
 * upsweep, spine, and downsweep kernels, this type includes enactor tuning
 * parameters that define kernel-dispatch policy.  By encapsulating the tuning information
 * for dispatch and both kernels, we assure operational consistency over an entire
 * BFS Compaction pass.
 */
template <
	// ProblemType type parameters
	typename _ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Common tunable params
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	int LOG_SCHEDULE_GRANULARITY,

	// Atomic expansion tunable params
	int ATOMIC_EXPAND_LOG_SCHEDULE_GRANULARITY,
	int ATOMIC_EXPAND_MAX_CTA_OCCUPANCY,
	int ATOMIC_EXPAND_LOG_THREADS,
	int ATOMIC_EXPAND_LOG_LOAD_VEC_SIZE,
	int ATOMIC_EXPAND_LOG_LOADS_PER_TILE,
	int ATOMIC_EXPAND_LOG_RAKING_THREADS,
	bool ATOMIC_EXPAND_WORK_STEALING,

	// Compaction upsweep tunable params
	int UPSWEEP_COMPACT_MAX_CTA_OCCUPANCY,
	int UPSWEEP_COMPACT_LOG_THREADS,
	int UPSWEEP_COMPACT_LOG_LOAD_VEC_SIZE,
	int UPSWEEP_COMPACT_LOG_LOADS_PER_TILE,

	// Compaction spine tunable params
	int SPINE_COMPACT_LOG_THREADS,
	int SPINE_COMPACT_LOG_LOAD_VEC_SIZE,
	int SPINE_COMPACT_LOG_LOADS_PER_TILE,
	int SPINE_COMPACT_LOG_RAKING_THREADS,

	// Compaction downsweep tunable params
	int DOWNSWEEP_COMPACT_MAX_CTA_OCCUPANCY,
	int DOWNSWEEP_COMPACT_LOG_THREADS,
	int DOWNSWEEP_COMPACT_LOG_LOAD_VEC_SIZE,
	int DOWNSWEEP_COMPACT_LOG_LOADS_PER_TILE,
	int DOWNSWEEP_COMPACT_LOG_RAKING_THREADS>

struct ProblemConfig : _ProblemType
{
	typedef _ProblemType ProblemType;
	typedef typename ProblemType::SizeT SizeT;

	//---------------------------------------------------------------------
	// Atomic Expansion
	//---------------------------------------------------------------------

	// Kernel config for the upsweep reduction kernel
	typedef expand_atomic::SweepKernelConfig <
		ProblemType,
		CUDA_ARCH,
		ATOMIC_EXPAND_MAX_CTA_OCCUPANCY,
		ATOMIC_EXPAND_LOG_THREADS,
		ATOMIC_EXPAND_LOG_LOAD_VEC_SIZE,
		ATOMIC_EXPAND_LOG_LOADS_PER_TILE,
		ATOMIC_EXPAND_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		ATOMIC_EXPAND_WORK_STEALING,
		ATOMIC_EXPAND_LOG_SCHEDULE_GRANULARITY>
			ExpandAtomicSweep;

	//---------------------------------------------------------------------
	// Compaction upsweep
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction upsweep kernel
	typedef compact::UpsweepKernelConfig <
		_ProblemType,
		CUDA_ARCH,
		UPSWEEP_COMPACT_MAX_CTA_OCCUPANCY,
		UPSWEEP_COMPACT_LOG_THREADS,
		UPSWEEP_COMPACT_LOG_LOAD_VEC_SIZE,
		UPSWEEP_COMPACT_LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			CompactUpsweep;

	//---------------------------------------------------------------------
	// Compaction spine
	//---------------------------------------------------------------------

	// Identity for spine
	static __device__ __forceinline__ SizeT CompactSpineIdentity() { return 0; }

	// Problem type for compaction spine
	typedef scan::ProblemType<
		SizeT,								// Spine-scan type is SizeT
		SizeT,								// Spine-scan sizet is SizeT
		true,								// Exclusive
		util::DefaultSum<SizeT>,
		CompactSpineIdentity> CompactSpineProblem;

	// Kernel config for the BFS compaction spine kernel
	typedef scan::DownsweepKernelConfig <
		CompactSpineProblem,
		CUDA_ARCH,
		1,									// Only a single-CTA grid
		SPINE_COMPACT_LOG_THREADS,
		SPINE_COMPACT_LOG_LOAD_VEC_SIZE,
		SPINE_COMPACT_LOG_LOADS_PER_TILE,
		SPINE_COMPACT_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		SPINE_COMPACT_LOG_LOADS_PER_TILE + SPINE_COMPACT_LOG_LOAD_VEC_SIZE + SPINE_COMPACT_LOG_THREADS>
			CompactSpine;

	//---------------------------------------------------------------------
	// Compaction downsweep
	//---------------------------------------------------------------------

	// Kernel config for the BFS compaction downsweep kernel
	typedef compact::DownsweepKernelConfig <
		_ProblemType,
		CUDA_ARCH,
		DOWNSWEEP_COMPACT_MAX_CTA_OCCUPANCY,
		DOWNSWEEP_COMPACT_LOG_THREADS,
		DOWNSWEEP_COMPACT_LOG_LOAD_VEC_SIZE,
		DOWNSWEEP_COMPACT_LOG_LOADS_PER_TILE,
		DOWNSWEEP_COMPACT_LOG_RAKING_THREADS,
		READ_MODIFIER,
		WRITE_MODIFIER,
		LOG_SCHEDULE_GRANULARITY>
			CompactDownsweep;




	enum {
		VALID = CompactUpsweep::VALID & CompactSpine::VALID & CompactDownsweep::VALID & ExpandAtomicSweep::VALID
	};

	static void Print()
	{
		printf("%s, %s, %d, "
				"%d, %d, %d, %d, "
				"%d, %d, %d, %d, "
				"%d, %d, %d, %d, %d",

			CacheModifierToString((int) READ_MODIFIER),
			CacheModifierToString((int) WRITE_MODIFIER),
			LOG_SCHEDULE_GRANULARITY,

			UPSWEEP_COMPACT_MAX_CTA_OCCUPANCY,
			UPSWEEP_COMPACT_LOG_THREADS,
			UPSWEEP_COMPACT_LOG_LOAD_VEC_SIZE,
			UPSWEEP_COMPACT_LOG_LOADS_PER_TILE,

			SPINE_COMPACT_LOG_THREADS,
			SPINE_COMPACT_LOG_LOAD_VEC_SIZE,
			SPINE_COMPACT_LOG_LOADS_PER_TILE,
			SPINE_COMPACT_LOG_RAKING_THREADS,

			DOWNSWEEP_COMPACT_MAX_CTA_OCCUPANCY,
			DOWNSWEEP_COMPACT_LOG_THREADS,
			DOWNSWEEP_COMPACT_LOG_LOAD_VEC_SIZE,
			DOWNSWEEP_COMPACT_LOG_LOADS_PER_TILE,
			DOWNSWEEP_COMPACT_LOG_RAKING_THREADS);
	}
};
		

} // namespace bfs
} // namespace b40c

