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
 * Upsweep reduction kernel for scan
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/reduction/cta.cuh>

namespace b40c {
namespace scan {


/**
 * Upsweep reduction pass
 */
template <typename KernelConfig, typename SmemStorage>
__device__ __forceinline__ void UpsweepPass(
	typename KernelConfig::T 									*&d_in,
	typename KernelConfig::T 									*&d_out,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	&work_decomposition,
	SmemStorage													&smem_storage)
{
	typedef reduction::Cta<KernelConfig>			Cta;
	typedef typename KernelConfig::SizeT 			SizeT;

	// Quit if we're the last threadblock (no need for it in upsweep)
	if (blockIdx.x == gridDim.x - 1) {
		return;
	}

	// CTA processing abstraction
	Cta cta(smem_storage, d_in, d_out);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelConfig::LOG_TILE_ELEMENTS,
		KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Since we're not the last block: process at least one full tile of tile_elements
	cta.template ProcessFullTile<true>(work_limits.offset);
	work_limits.offset += KernelConfig::TILE_ELEMENTS;

	// Process any other full tiles
	while (work_limits.offset < work_limits.guarded_offset) {

		cta.ProcessFullTile<false>(work_limits.offset);
		work_limits.offset += KernelConfig::TILE_ELEMENTS;
	}

	// Collectively reduce accumulated carry from each thread into output
	// destination (all thread have valid reduction partials)
	cta.OutputToSpine();
}


/******************************************************************************
 * Upsweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Upsweep reduction kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void UpsweepKernel(
	typename KernelConfig::T 									*d_in,
	typename KernelConfig::T 									*d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	UpsweepPass<KernelConfig>(d_in, d_spine, work_decomposition, smem_storage);
}


} // namespace scan
} // namespace b40c

