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
 * Upsweep BFS Compaction kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/bfs/compact/upsweep_cta.cuh>

namespace b40c {
namespace bfs {
namespace compact {


/**
 * Upsweep BFS Compaction pass
 */
template <typename KernelConfig, typename SmemStorage>
__device__ __forceinline__ void UpsweepPass(
	typename KernelConfig::VertexId 				*&d_in,
	typename KernelConfig::ValidFlag				*&d_out_flag,
	typename KernelConfig::SizeT 					*&d_spine,
	typename KernelConfig::CollisionMask 			*&d_collision_cache,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
	SmemStorage										&smem_storage)
{
	typedef UpsweepCta<KernelConfig, SmemStorage> 	UpsweepCta;
	typedef typename KernelConfig::SizeT 			SizeT;

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
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

	// Process full tiles
	while (work_limits.offset < work_limits.guarded_offset) {
		cta.ProcessFullTile(work_limits.offset);
		work_limits.offset += KernelConfig::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (work_limits.guarded_elements) {
		cta.ProcessPartialTile(
			work_limits.offset,
			work_limits.out_of_bounds);
	}

	// Collectively reduce accumulated carry from each thread into output
	// destination (all thread have valid reduction partials)
	cta.OutputToSpine();
}


/******************************************************************************
 * Upsweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Upsweep BFS Compaction kernel entry point
 */
template <typename KernelConfig, bool INSTRUMENT>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	typename KernelConfig::VertexId			iteration,
	volatile int							*d_done,
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

		// Signal to host that we're done
		if (num_elements == 0) {
			d_done[0] = 1;
		}

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
		smem_storage.work_decomposition,
		smem_storage);

	if (INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStop();
		}
	}
}


} // namespace compact
} // namespace bfs
} // namespace b40c

