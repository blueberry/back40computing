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
 * Consecutive removal downsweep kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/srts_details.cuh>
#include <b40c/consecutive_removal/downsweep_cta.cuh>

namespace b40c {
namespace consecutive_removal {


/**
 * Downsweep consecutive removal pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void DownsweepPass(
	typename KernelConfig::T 			* &d_in,
	typename KernelConfig::SizeT		* &d_num_compacted,
	typename KernelConfig::T 			* &d_out,
	typename KernelConfig::FlagCount	* &d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition)
{
	typedef typename KernelConfig::SrtsDetails 		SrtsDetails;
	typedef DownsweepCta<KernelConfig> 				DownsweepCta;
	typedef typename KernelConfig::T 				T;
	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::FlagCount 		FlagCount;			// Type for discontinuity counts
	typedef typename KernelConfig::LocalFlagCount	LocalFlagCount;		// Type for local discontinuity counts (just needs to count up to TILE_ELEMENTS)

	// Shared storage for CTA processing
	__shared__ uint4 smem_pool[KernelConfig::SMEM_POOL_QUADS];
	__shared__ LocalFlagCount warpscan[2][B40C_WARP_THREADS(KernelConfig::CUDA_ARCH)];

	// SRTS grid details
	SrtsDetails srts_detail(
		smem_pool,
		warpscan,
		0);					// Scan identity for discontinuity counts

	// We need the exclusive partial from our spine
	FlagCount spine_partial = 0;
	if (d_spine != NULL) {
		util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
			spine_partial, d_spine + blockIdx.x);
	}

	// CTA processing abstraction
	DownsweepCta cta(
		srts_detail,
		reinterpret_cast<T*>(smem_pool),
		d_in,
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

	if (cta_offset < guarded_offset) {

		// Process at least one full tile of tile_elements (first tile)
		cta.template ProcessTile<true, true>(cta_offset, out_of_bounds);
		cta_offset += KernelConfig::TILE_ELEMENTS;

		while (cta_offset < guarded_offset) {
			// Process more full tiles (not first tile)
			cta.template ProcessTile<true, false>(cta_offset, out_of_bounds);
			cta_offset += KernelConfig::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io (not first tile)
		if (guarded_elements) {
			cta.template ProcessTile<false, false>(cta_offset, out_of_bounds);
		}

	} else {

		// Clean up last partial tile with guarded-io (first tile)
		cta.template ProcessTile<false, true>(cta_offset, out_of_bounds);
	}

	if ((blockIdx.x == gridDim.x - 1) && (threadIdx.x == 0)) {
		util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
			cta.carry, d_num_compacted);
	}
}


/******************************************************************************
 * Downsweep Consecutive Removal Kernel Entrypoint
 ******************************************************************************/

/**
 * Downsweep consecutive removal kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void DownsweepKernel(
	typename KernelConfig::T 			* d_in,
	typename KernelConfig::SizeT		* d_num_compacted,
	typename KernelConfig::T 			* d_out,
	typename KernelConfig::FlagCount	* d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition)
{
	DownsweepPass<KernelConfig>(d_in, d_num_compacted, d_out, d_spine, work_decomposition);
}


} // namespace consecutive_removal
} // namespace b40c

