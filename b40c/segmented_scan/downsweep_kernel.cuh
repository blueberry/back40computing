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

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/segmented_scan/downsweep_cta.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Downsweep scan pass
 */
template <typename KernelConfig, typename SmemStorage>
__device__ __forceinline__ void DownsweepPass(
	typename KernelConfig::T 			*&d_partials_in,
	typename KernelConfig::Flag			*&d_flags_in,
	typename KernelConfig::T 			*&d_partials_out,
	typename KernelConfig::T 			*&d_spine_partials,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
	SmemStorage							&smem_storage)
{
	typedef DownsweepCta<KernelConfig> 		DownsweepCta;
	typedef typename KernelConfig::T		T;
	typedef typename KernelConfig::SizeT 	SizeT;


	// Read the exclusive partial from our spine
	T spine_partial;
	util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
		spine_partial, d_spine_partials + blockIdx.x);

	// CTA processing abstraction
	DownsweepCta cta(
		smem_storage,
		d_partials_in,
		d_flags_in,
		d_partials_out,
		spine_partial);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelConfig::LOG_TILE_ELEMENTS,
		KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Process full tiles of tile_elements
	while (work_limits.offset < work_limits.guarded_offset) {

		cta.template ProcessTile<true>(work_limits.offset);
		work_limits.offset += KernelConfig::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (work_limits.guarded_elements) {
		cta.template ProcessTile<false>(
			work_limits.offset,
			work_limits.guarded_elements);
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
void DownsweepKernel(
	typename KernelConfig::T 									*d_partials_in,
	typename KernelConfig::Flag									*d_flags_in,
	typename KernelConfig::T 									*d_partials_out,
	typename KernelConfig::T 									*d_spine_partials,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	DownsweepPass<KernelConfig>(
		d_partials_in,
		d_flags_in,
		d_partials_out,
		d_spine_partials,
		work_decomposition,
		smem_storage);
}


} // namespace segmented_scan
} // namespace b40c

