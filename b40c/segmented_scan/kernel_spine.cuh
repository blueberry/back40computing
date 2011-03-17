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
 * Segmented scan spine kernel
 ******************************************************************************/

#pragma once

#include <b40c/segmented_scan/scan_cta.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Segmented scan spine pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void SpineScanPass(
	typename KernelConfig::T 		*&d_partials_in,
	typename KernelConfig::Flag		*&d_flags_in,
	typename KernelConfig::T 		*&d_partials_out,
	typename KernelConfig::SizeT 	&spine_elements)
{
	typedef ScanCta<KernelConfig> ScanCta;
	typedef typename ScanCta::T T;
	typedef typename ScanCta::Flag Flag;
	typedef typename ScanCta::SizeT SizeT;
	typedef typename ScanCta::SrtsSoaDetails SrtsSoaDetails;

	// Exit if we're not the first CTA
	if (blockIdx.x > 0) return;

	// Shared SRTS grid storage
	__shared__ uint4 partial_smem_pool[KernelConfig::PartialsSrtsGrid::SMEM_QUADS];
	__shared__ uint4 flag_smem_pool[KernelConfig::FlagsSrtsGrid::SMEM_QUADS];

	// Shared SRTS warpscan storage
	__shared__ T partials_warpscan[2][B40C_WARP_THREADS(KernelConfig::CUDA_ARCH)];
	__shared__ Flag flags_warpscan[2][B40C_WARP_THREADS(KernelConfig::CUDA_ARCH)];

	// SRTS grid details
	SrtsSoaDetails srts_soa_details(
		typename SrtsSoaDetails::GridStorageSoa(partial_smem_pool, flag_smem_pool),
		typename SrtsSoaDetails::WarpscanSoa(partials_warpscan, flags_warpscan));

	// CTA processing abstraction
	ScanCta cta(
		srts_soa_details,
		d_partials_in,
		d_flags_in,
		d_partials_out);

	// Number of elements in (the last) partially-full tile (requires guarded loads)
	SizeT cta_guarded_elements = spine_elements & (ScanCta::TILE_ELEMENTS - 1);

	// Offset of final, partially-full tile (requires guarded loads)
	SizeT cta_guarded_offset = spine_elements - cta_guarded_elements;

	// Process full tiles of tile_elements
	SizeT cta_offset = 0;
	while (cta_offset < cta_guarded_offset) {

		cta.ProcessTile<true>(cta_offset, cta_guarded_offset);
		cta_offset += ScanCta::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (cta_guarded_elements) {
		cta.ProcessTile<false>(cta_offset, spine_elements);
	}
}


/******************************************************************************
 * Spine Scan Kernel Entry-point
 ******************************************************************************/

/**
 * Spine scan kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void SpineScanKernel(
	typename KernelConfig::T 		*d_partials_in,
	typename KernelConfig::Flag		*d_flags_in,
	typename KernelConfig::T 		*d_partials_out,
	typename KernelConfig::SizeT 	spine_elements)
{
	SpineScanPass<KernelConfig>(
		d_partials_in, d_flags_in, d_partials_out, spine_elements);
}


} // namespace segmented_scan
} // namespace b40c

