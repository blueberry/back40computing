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
 * Reduction spine kernel
 ******************************************************************************/

#pragma once

#include <b40c/reduction/reduction_cta.cuh>

namespace b40c {
namespace reduction {


/**
 * Spine reduction pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void SpineReductionPass(
	typename KernelConfig::T 		*d_in,
	typename KernelConfig::T 		*d_out,
	typename KernelConfig::SizeT 	spine_elements)
{
	typedef ReductionCta<KernelConfig> ReductionCta;
	typedef typename ReductionCta::T T;
	typedef typename ReductionCta::SizeT SizeT;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// Shared SRTS grid storage
	__shared__ uint4 reduction_tree[KernelConfig::SMEM_QUADS];

	// CTA processing abstraction
	ReductionCta cta(reduction_tree, d_in, d_out);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT guarded_elements = spine_elements & (ReductionCta::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_offset = spine_elements - guarded_elements;

	// Process tiles of tile_elements
	SizeT cta_offset = 0;
	if (cta_offset < guarded_offset) {

		// Process at least one full tile of tile_elements

		cta.ProcessFullTile<true>(cta_offset, spine_elements);
		cta_offset += ReductionCta::TILE_ELEMENTS;

		while (cta_offset < guarded_offset) {

			cta.ProcessFullTile<false>(cta_offset, spine_elements);
			cta_offset += ReductionCta::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (guarded_elements) {
			cta.ProcessPartialTile<false>(cta_offset, spine_elements);
		}

		// Collectively reduce accumulated carry from each thread into output
		// destination (all thread have valid reduction partials)
		cta.template FinalReduction<true>(KernelConfig::THREADS);

	} else {

		// Clean up last partial tile with guarded-io
		cta.ProcessPartialTile<true>(cta_offset, spine_elements);

		// Collectively reduce accumulated carry from each thread into output
		// destination (not every thread may have a valid reduction partial)
		cta.template FinalReduction<false>(spine_elements);
	}

}


/******************************************************************************
 * Spine Reduction Kernel Entry-point
 ******************************************************************************/

/**
 * Spine reduction kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void SpineReductionKernel(
	typename KernelConfig::T 		*d_in,
	typename KernelConfig::T 		*d_out,
	typename KernelConfig::SizeT 	spine_elements)
{
	SpineReductionPass<KernelConfig>(d_in, d_out, spine_elements);
}


} // namespace reduction
} // namespace b40c

