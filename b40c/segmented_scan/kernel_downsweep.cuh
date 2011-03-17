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
 * Scan kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/work_distribution.cuh>
#include <b40c/segmented_scan/scan_cta.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Downsweep scan pass
 */
template <typename KernelConfig>
__device__ __forceinline__ void DownsweepScanPass(
	typename KernelConfig::T 			*&d_partials_in,
	typename KernelConfig::Flag			*&d_flags_in,
	typename KernelConfig::T 			*&d_partials_out,
	typename KernelConfig::T 			*&d_spine_partials,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition)
{
	typedef ScanCta<KernelConfig> ScanCta;
	typedef typename ScanCta::T T;
	typedef typename ScanCta::Flag Flag;
	typedef typename ScanCta::SizeT SizeT;
	typedef typename ScanCta::SrtsSoaDetails SrtsSoaDetails;

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

	// Read the exclusive partial from our spine
	T spine_partial;
	util::ModifiedLoad<T, KernelConfig::READ_MODIFIER>::Ld(
		spine_partial, d_spine_partials, blockIdx.x);

	// CTA processing abstraction
	ScanCta cta(
		srts_soa_details,
		d_partials_in,
		d_flags_in,
		d_partials_out,
		spine_partial);

	// Determine our threadblock's work range
	SizeT cta_offset;			// Offset at which this CTA begins processing
	SizeT cta_elements;			// Total number of elements for this CTA to process
	SizeT guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	SizeT guarded_elements;		// Number of elements in partially-full tile

	work_decomposition.GetCtaWorkLimits<ScanCta::LOG_TILE_ELEMENTS, ScanCta::LOG_SCHEDULE_GRANULARITY>(
		cta_offset, cta_elements, guarded_offset, guarded_elements);

	SizeT out_of_bounds = cta_offset + cta_elements;

	// Process full tiles of tile_elements
	while (cta_offset < guarded_offset) {

		cta.ProcessTile<true>(cta_offset, out_of_bounds);
		cta_offset += ScanCta::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (guarded_elements) {
		cta.ProcessTile<false>(cta_offset, out_of_bounds);
	}
}



/******************************************************************************
 * Downsweep Scan Kernel Entrypoint
 ******************************************************************************/

/**
 * Downsweep scan kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void DownsweepScanKernel(
	typename KernelConfig::T 									*d_partials_in,
	typename KernelConfig::Flag									*d_flags_in,
	typename KernelConfig::T 									*d_partials_out,
	typename KernelConfig::T 									*d_spine_partials,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	work_decomposition)
{
	DownsweepScanPass<KernelConfig>(
		d_partials_in,
		d_flags_in,
		d_partials_out,
		d_spine_partials,
		work_decomposition);
}


} // namespace segmented_scan
} // namespace b40c

