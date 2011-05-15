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
 * Spine kernel
 ******************************************************************************/

#pragma once

#include <b40c/scan/downsweep/cta.cuh>

namespace b40c {
namespace scan {
namespace spine {


/**
 * Spine scan pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void SpinePass(
	typename KernelPolicy::T 				*&d_in,
	typename KernelPolicy::T 				*&d_out,
	typename KernelPolicy::SizeT 			&spine_elements,
	typename KernelPolicy::SmemStorage		&smem_storage)
{
	typedef downsweep::Cta<KernelPolicy> 		Cta;
	typedef typename KernelPolicy::SizeT 		SizeT;
	typedef typename KernelPolicy::T 			T;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// CTA processing abstraction
	Cta cta(smem_storage, d_in, d_out);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT cta_guarded_elements = spine_elements & (KernelPolicy::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT cta_guarded_offset = spine_elements - cta_guarded_elements;

	// Process full tiles of tile_elements
	SizeT cta_offset = 0;
	while (cta_offset < cta_guarded_offset) {
		cta.ProcessTile(cta_offset);
		cta_offset += KernelPolicy::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (cta_guarded_elements) {
		cta.ProcessTile(cta_offset, spine_elements);
	}
}


/**
 * Spine scan kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__ 
void Kernel(
	typename KernelPolicy::T			*d_in,
	typename KernelPolicy::T			*d_out,
	typename KernelPolicy::SizeT 		spine_elements)
{
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	SpinePass<KernelPolicy>(d_in, d_out, spine_elements, smem_storage);
}

} // namespace spine
} // namespace scan
} // namespace b40c

