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
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Upsweep kernel (bin reduction/counting)
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/bfs/partition_compact/upsweep/cta.cuh>

namespace b40c {
namespace bfs {
namespace partition_compact {
namespace upsweep {


/**
 * Upsweep compaction pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void UpsweepPass(
	typename KernelConfig::VertexId 				*&d_in,
	typename KernelConfig::ValidFlag				*&d_out_flag,
	typename KernelConfig::SizeT 					*&d_spine,
	typename KernelConfig::CollisionMask 			*&d_collision_cache,
	typename KernelConfig::SmemStorage				&smem_storage)
{
	typedef Cta<KernelConfig, SmemStorage> 	UpsweepCta;
	typedef typename KernelConfig::SizeT 			SizeT;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	smem_storage.work_decomposition.template GetCtaWorkLimits<
		KernelConfig::LOG_TILE_ELEMENTS,
		KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Return if we have no work to do
	if (!work_limits.elements) {
		return;
	}

	// CTA processing abstraction
	UpsweepCta cta(
		smem_storage,
		d_in,
		d_out_flag,
		d_spine,
		d_collision_cache);

	// Process all tiles
	cta.ProcessTiles(work_limits.offset, work_limits.out_of_bounds);
}


/**
 * Upsweep compaction kernel entry point
 */
template <typename KernelConfig, bool INSTRUMENT>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	typename KernelConfig::VertexId			iteration,
	typename KernelConfig::VertexId 		*d_in,
	typename KernelConfig::ValidFlag		*d_out_flag,
	typename KernelConfig::SizeT			*d_spine,
	typename KernelConfig::CollisionMask 	*d_collision_cache,
	util::CtaWorkProgress 					work_progress,
	util::KernelRuntimeStats				kernel_stats)
{
	typedef typename KernelConfig::SizeT SizeT;

	// Shared storage for CTA processing
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	if (INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStart();
		}
	}

	// Determine work decomposition
	if (threadIdx.x == 0) {
		// Obtain problem size
		SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(iteration);

		// Initialize work decomposition in smem
		smem_storage.work_decomposition.template Init<KernelConfig::LOG_SCHEDULE_GRANULARITY>(
			num_elements, gridDim.x);
	}

	// Barrier to protect work decomposition
	__syncthreads();

	UpsweepPass<KernelConfig>(
		d_in,
		d_out_flag,
		d_spine,
		d_collision_cache,
		smem_storage);

	if (INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStop();
		}
	}
}


} // namespace upsweep
} // namespace partition_compact
} // namespace bfs
} // namespace b40c

