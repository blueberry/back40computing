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
 * Downsweep kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/srts_details.cuh>

#include <b40c/scan/downsweep/cta.cuh>

namespace b40c {
namespace scan {
namespace downsweep {


/**
 * Downsweep scan pass
 */
template <typename KernelPolicy>
__device__ __forceinline__ void DownsweepPass(
	typename KernelPolicy::T 									* &d_in,
	typename KernelPolicy::T 									* &d_out,
	typename KernelPolicy::T 									* &d_spine,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> 	&work_decomposition,
	typename KernelPolicy::SmemStorage							&smem_storage)
{
	typedef Cta<KernelPolicy> 				Cta;
	typedef typename KernelPolicy::T 		T;
	typedef typename KernelPolicy::SizeT 	SizeT;

	// Obtain exclusive spine partial
	T spine_partial;
	util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(
		spine_partial, d_spine + blockIdx.x);

	// CTA processing abstraction
	Cta cta(smem_storage, d_in, d_out, spine_partial);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelPolicy::LOG_TILE_ELEMENTS,
		KernelPolicy::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Process full tiles of tile_elements
	while (work_limits.offset < work_limits.guarded_offset) {

		cta.ProcessTile(work_limits.offset);
		work_limits.offset += KernelPolicy::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (work_limits.guarded_elements) {
		cta.ProcessTile(
			work_limits.offset,
			work_limits.guarded_elements);
	}
}


/**
 * Downsweep scan kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::T 			* d_in,
	typename KernelPolicy::T 			* d_out,
	typename KernelPolicy::T 			* d_spine,
	util::CtaWorkDistribution<typename KernelPolicy::SizeT> work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	DownsweepPass<KernelPolicy>(
		d_in,
		d_out,
		d_spine,
		work_decomposition,
		smem_storage);
}

} // namespace downsweep
} // namespace scan
} // namespace b40c

