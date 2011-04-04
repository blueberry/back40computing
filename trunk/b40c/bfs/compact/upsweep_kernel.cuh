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
#include <b40c/bfs/compact/upsweep_cta.cuh>

namespace b40c {
namespace bfs {
namespace compact {


/**
 * Upsweep BFS Compaction pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void UpsweepPass(
	typename KernelConfig::VertexId 							*&d_in,
	unsigned char												*&d_out_flag,
	typename KernelConfig::SizeT 								*&d_spine,
	unsigned char 												*&d_collision_cache,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	&work_decomposition)
{
	typedef UpsweepSmemStorage<KernelConfig> 		SmemStorage;
	typedef UpsweepCta<KernelConfig, SmemStorage> 	UpsweepCta;
	typedef typename KernelConfig::VertexId 		VertexId;
	typedef typename KernelConfig::SizeT 			SizeT;

	// Shared storage
	__shared__ SmemStorage smem_storage;

	// CTA processing abstraction
	UpsweepCta cta(
		smem_storage,
		d_in,
		d_out_flag,
		d_spine,
		d_collision_cache);

	// Determine our threadblock's work range
	SizeT cta_offset;			// Offset at which this CTA begins processing
	SizeT cta_elements;			// Total number of elements for this CTA to process
	SizeT guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_elements;		// Number of elements in partially-full tile

	work_decomposition.template GetCtaWorkLimits<KernelConfig::LOG_TILE_ELEMENTS, KernelConfig::LOG_SCHEDULE_GRANULARITY>(
		cta_offset, cta_elements, guarded_offset, guarded_elements);

	SizeT out_of_bounds = cta_offset + cta_elements;

	// Process full tiles
	while (cta_offset < guarded_offset) {
		cta.ProcessFullTile(cta_offset, out_of_bounds);
		cta_offset += KernelConfig::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io (not first tile)
	if (guarded_elements) {
		cta.ProcessPartialTile(cta_offset, out_of_bounds);
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
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	typename KernelConfig::VertexId 							*d_in,
	unsigned char												*d_out_flag,
	typename KernelConfig::SizeT								*d_spine,
	unsigned char 												*d_collision_cache,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	work_decomposition)
{
	UpsweepPass<KernelConfig>(
		d_in,
		d_out_flag,
		d_spine,
		d_collision_cache,
		work_decomposition);
}


} // namespace compact
} // namespace bfs
} // namespace b40c

