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
#include <b40c/util/srts_details.cuh>
#include <b40c/scan/cta.cuh>

namespace b40c {
namespace scan {


/**
 * Downsweep scan pass
 */
template <typename KernelConfig, typename SmemStorage>
__device__ __forceinline__ void DownsweepPass(
	typename KernelConfig::T 			* &d_in,
	typename KernelConfig::T 			* &d_out,
	typename KernelConfig::T 			* &d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
	SmemStorage							&smem_storage)
{
	typedef Cta<KernelConfig> 				Cta;
	typedef typename KernelConfig::T 		T;
	typedef typename KernelConfig::SizeT 	SizeT;

	// We need the exclusive partial from our spine, regardless of whether
	// we're exclusive/inclusive

	T spine_partial;
	if (KernelConfig::EXCLUSIVE) {

		// Spine was an exclusive scan
		util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
			spine_partial, d_spine + blockIdx.x);

	} else {

		// Spine was in inclusive scan: load exclusive partial
		if (blockIdx.x == 0) {
			spine_partial = KernelConfig::Identity();
		} else {
			util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
				spine_partial, d_spine + blockIdx.x - 1);
		}
	}

	// CTA processing abstraction
	Cta cta(smem_storage, d_in, d_out, spine_partial);

	// Determine our threadblock's work range
	util::CtaWorkLimits<SizeT> work_limits;
	work_decomposition.template GetCtaWorkLimits<
		KernelConfig::LOG_TILE_ELEMENTS,
		KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

	// Process full tiles of tile_elements
	while (work_limits.offset < work_limits.guarded_offset) {

		cta.ProcessTile(work_limits.offset);
		work_limits.offset += KernelConfig::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (work_limits.guarded_elements) {
		cta.ProcessTile(
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
	typename KernelConfig::T 			* d_in,
	typename KernelConfig::T 			* d_out,
	typename KernelConfig::T 			* d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition)
{
	// Shared storage for the kernel
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	DownsweepPass<KernelConfig>(
		d_in,
		d_out,
		d_spine,
		work_decomposition,
		smem_storage);
}


} // namespace scan
} // namespace b40c

