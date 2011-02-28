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

#include <b40c/reduction/kernel_tile.cuh>

namespace b40c {
namespace reduction {


/**
 * Spine reduction pass
 */
template <typename ReductionKernelConfig>
__device__ __forceinline__ void SpineReductionPass(
	typename ReductionKernelConfig::T 		* __restrict 	d_spine,
	typename ReductionKernelConfig::T 		* __restrict 	d_out,
	typename ReductionKernelConfig::SizeT 					spine_elements)
{
	typedef ReductionTile<ReductionKernelConfig> Tile;
	typedef typename Tile::SizeT SizeT;
	typedef typename Tile::T T;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// The value we will accumulate
	T carry = Tile::Identity();

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT cta_guarded_elements = spine_elements & (Tile::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT cta_guarded_offset = spine_elements - cta_guarded_elements;

	// Process full tiles of tile_elements
	SizeT cta_offset = 0;
	while (cta_offset < cta_guarded_offset) {

		Tile::ProcessTile<true>(d_spine, cta_offset, cta_guarded_offset, carry);
		cta_offset += Tile::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (cta_guarded_elements) {
		Tile::ProcessTile<false>(d_spine, cta_offset, spine_elements, carry);
	}

	// Collectively reduce accumulated carry from each thread
	Tile::CollectiveReduction(carry, d_out);
}


/******************************************************************************
 * Spine Reduction Kernel Entry-point
 ******************************************************************************/

/**
 * Spine reduction kernel entry point
 */
template <typename ReductionKernelConfig>
__launch_bounds__ (ReductionKernelConfig::THREADS, ReductionKernelConfig::CTA_OCCUPANCY)
__global__ 
void SpineReductionKernel(
	typename ReductionKernelConfig::T 		* __restrict 	d_spine,
	typename ReductionKernelConfig::T 		* __restrict 	d_out,
	typename ReductionKernelConfig::SizeT 					spine_elements)
{
	SpineReductionPass<ReductionKernelConfig>(d_spine, d_out, spine_elements);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename ReductionKernelConfig>
void __wrapper__device_stub_SpineReductionKernel(
		typename ReductionKernelConfig::T * __restrict &,
		typename ReductionKernelConfig::T * __restrict &,
		typename ReductionKernelConfig::SizeT&) {}




} // namespace reduction
} // namespace b40c

