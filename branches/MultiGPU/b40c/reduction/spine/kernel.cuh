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
 * Spine reduction kernel
 ******************************************************************************/

#pragma once

#include <b40c/reduction/cta.cuh>

namespace b40c {
namespace reduction {
namespace spine {


/**
 * Spine reduction pass
 */
template <typename KernelPolicy, typename SmemStorage>
__device__ __forceinline__ void SpinePass(
	typename KernelPolicy::T 		*d_in,
	typename KernelPolicy::T 		*d_spine,
	typename KernelPolicy::SizeT 	spine_elements,
	SmemStorage						&smem_storage)
{
	typedef Cta<KernelPolicy> 				Cta;
	typedef typename KernelPolicy::T 		T;
	typedef typename KernelPolicy::SizeT 	SizeT;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// CTA processing abstraction
	Cta cta(smem_storage, d_in, d_spine);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT guarded_elements = spine_elements & (KernelPolicy::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_offset = spine_elements - guarded_elements;

	// Process tiles of tile_elements
	SizeT cta_offset = 0;
	if (cta_offset < guarded_offset) {

		// Process at least one full tile of tile_elements
		cta.ProcessFullTile<true>(cta_offset);
		cta_offset += KernelPolicy::TILE_ELEMENTS;

		// Process more full tiles (not first tile)
		while (cta_offset < guarded_offset) {
			cta.ProcessFullTile<false>(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (guarded_elements) {
			cta.ProcessPartialTile<false>(cta_offset, spine_elements);
		}

		// Collectively reduce accumulated carry from each thread into output
		// destination (all thread have valid reduction partials)
		cta.OutputToSpine();

	} else {

		// Clean up last partial tile with guarded-io
		cta.ProcessPartialTile<true>(cta_offset, spine_elements);

		// Collectively reduce accumulated carry from each thread into output
		// destination (not every thread may have a valid reduction partial)
		cta.OutputToSpine(spine_elements);
	}
}


/**
 * Spine reduction kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::T 		*d_in,
	typename KernelPolicy::T 		*d_spine,
	typename KernelPolicy::SizeT 	spine_elements)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	SpinePass<KernelPolicy>(d_in, d_spine, spine_elements, smem_storage);
}

} // namespace spine
} // namespace reduction
} // namespace b40c

