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
 * Reduction kernel
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
	typedef typename ReductionCta::SizeT SizeT;
	typedef typename ReductionCta::T T;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// Shared storage for CTA processing
	__shared__ uint4 smem_pool[KernelConfig::SRTS_GRID_QUADS];
	__shared__ T warpscan[2][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	// CTA processing abstraction
	ReductionCta cta(smem_pool, warpscan, d_in, d_out);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT cta_guarded_elements = spine_elements & (ReductionCta::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT cta_guarded_offset = spine_elements - cta_guarded_elements;

	// Process full tiles of tile_elements
	SizeT cta_offset = 0;
	while (cta_offset < cta_guarded_offset) {

		cta.ProcessTile<true>(cta_offset, cta_guarded_offset);
		cta_offset += ReductionCta::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (cta_guarded_elements) {
		cta.ProcessTile<false>(cta_offset, spine_elements);
	}

	// Collectively reduce accumulated carry from each thread
	cta.FinalReduction();
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


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
/*
template <typename KernelConfig>
void __wrapper__device_stub_SpineReductionKernel(
		typename KernelConfig::T *&,
		typename KernelConfig::T *&,
		typename KernelConfig::SizeT&) {}
*/



} // namespace reduction
} // namespace b40c

