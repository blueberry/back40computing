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
 * BFS Compaction downsweep kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/bfs/compact/downsweep_cta.cuh>

namespace b40c {
namespace bfs {
namespace compact {


/**
 * Downsweep BFS Compaction pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void DownsweepPass(
	typename KernelConfig::VertexId 			* &d_in,
	unsigned char								* &d_flags_in,
	typename KernelConfig::SizeT				* &d_num_compacted,
	typename KernelConfig::VertexId 			* &d_out,
	typename KernelConfig::SizeT 				* &d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition)
{
	typedef DownsweepSmemStorage<KernelConfig>			SmemStorage;
	typedef DownsweepCta<KernelConfig, SmemStorage> 	DownsweepCta;
	typedef typename KernelConfig::VertexId 			VertexId;
	typedef typename KernelConfig::SizeT 				SizeT;

	// Shared storage for CTA processing
	__shared__ SmemStorage smem_storage;

	// We need the exclusive partial from our spine
	SizeT spine_partial;
	util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
		spine_partial, d_spine + blockIdx.x);

	// CTA processing abstraction
	DownsweepCta cta(
		smem_storage,
		d_in,
		d_flags_in,
		d_out,
		spine_partial);

	// Determine our threadblock's work range
	SizeT cta_offset;			// Offset at which this CTA begins processing
	SizeT cta_elements;			// Total number of elements for this CTA to process
	SizeT guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_elements;		// Number of elements in partially-full tile

	work_decomposition.GetCtaWorkLimits<DownsweepCta::LOG_TILE_ELEMENTS, DownsweepCta::LOG_SCHEDULE_GRANULARITY>(
		cta_offset, cta_elements, guarded_offset, guarded_elements);

	SizeT out_of_bounds = cta_offset + cta_elements;

	// Process full tiles
	while (cta_offset < guarded_offset) {
		cta.template ProcessTile<true>(cta_offset, out_of_bounds);
		cta_offset += KernelConfig::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io (not first tile)
	if (guarded_elements) {
		cta.template ProcessTile<false>(cta_offset, out_of_bounds);
	}

	// Write out compacted length
	if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == 0)) {
		util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
			cta.carry, d_num_compacted);
	}
}


/******************************************************************************
 * Downsweep BFS Compaction Kernel Entrypoint
 ******************************************************************************/

/**
 * Downsweep BFS Compaction kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void DownsweepKernel(
	typename KernelConfig::VertexId 			* d_in,
	unsigned char								* d_flags_in,
	typename KernelConfig::SizeT				* d_num_compacted,
	typename KernelConfig::VertexId 			* d_out,
	typename KernelConfig::SizeT				* d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition)
{
	DownsweepPass<KernelConfig>(d_in, d_flags_in, d_num_compacted, d_out, d_spine, work_decomposition);
}


} // namespace compact
} // namespace bfs
} // namespace b40c

