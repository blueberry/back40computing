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

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/reduction/cta.cuh>

namespace b40c {
namespace reduction {



/**
 * Upsweep reduction pass (non-workstealing)
 */
template <typename KernelConfig, bool WORK_STEALING>
struct UpsweepPass
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelConfig::T 									*&d_in,
		typename KernelConfig::T 									*&d_out,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> 	&work_decomposition,
		util::CtaWorkProgress 										&work_progress,
		SmemStorage													&smem_storage)
	{
		typedef Cta<KernelConfig> 				Cta;
		typedef typename KernelConfig::SizeT 	SizeT;

		// CTA processing abstraction
		Cta cta(smem_storage, d_in, d_out);

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelConfig::LOG_TILE_ELEMENTS,
			KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

		if (work_limits.offset < work_limits.guarded_offset) {

			// Process at least one full tile of tile_elements
			cta.template ProcessFullTile<true>(work_limits.offset);
			work_limits.offset += KernelConfig::TILE_ELEMENTS;

			// Process more full tiles (not first tile)
			while (work_limits.offset < work_limits.guarded_offset) {
				cta.template ProcessFullTile<false>(work_limits.offset);
				work_limits.offset += KernelConfig::TILE_ELEMENTS;
			}

			// Clean up last partial tile with guarded-io (not first tile)
			if (work_limits.guarded_elements) {
				cta.template ProcessPartialTile<false>(
					work_limits.offset,
					work_limits.out_of_bounds);
			}

			// Collectively reduce accumulated carry from each thread into output
			// destination (all thread have valid reduction partials)
			cta.OutputToSpine();

		} else {

			// Clean up last partial tile with guarded-io (first tile)
			cta.template ProcessPartialTile<true>(
				work_limits.offset,
				work_limits.out_of_bounds);

			// Collectively reduce accumulated carry from each thread into output
			// destination (not every thread may have a valid reduction partial)
			cta.OutputToSpine(work_limits.elements);
		}

	}
};


template <typename SizeT>
__device__ __forceinline__ SizeT StealWork(
	util::CtaWorkProgress &work_progress,
	int count)
{
	__shared__ SizeT s_offset;		// The offset at which this CTA performs tile processing, shared by all

	// Thread zero atomically steals work from the progress counter
	if (threadIdx.x == 0) {
		s_offset = work_progress.Steal<SizeT>(count);
	}

	__syncthreads();		// Protect offset

	return s_offset;
}



/**
 * Upsweep reduction pass (workstealing)
 */
template <typename KernelConfig>
struct UpsweepPass <KernelConfig, true>
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelConfig::T 									*&d_in,
		typename KernelConfig::T 									*&d_out,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> 	&work_decomposition,
		util::CtaWorkProgress 										&work_progress,
		SmemStorage													&smem_storage)
	{
		typedef Cta<KernelConfig> 				Cta;
		typedef typename KernelConfig::SizeT 	SizeT;

		// CTA processing abstraction
		Cta cta(smem_storage, d_in, d_out);

		// First CTA resets the work progress for the next pass
		if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
			work_progress.template PrepResetSteal<SizeT>();
		}

		// Total number of elements in full tiles
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelConfig::TILE_ELEMENTS - 1));

		// Each CTA needs to process at least one partial block of
		// input (otherwise our spine scan will be invalid)

		SizeT offset = blockIdx.x << KernelConfig::LOG_TILE_ELEMENTS;
		if (offset < unguarded_elements) {

			// Process our one full tile (first tile seen)
			cta.template ProcessFullTile<true>(offset);

			// Determine the swath we just did
			SizeT swath = work_decomposition.grid_size << KernelConfig::LOG_TILE_ELEMENTS;

			// Worksteal subsequent full tiles, if any
			while ((offset = StealWork<SizeT>(
				work_progress,
				KernelConfig::TILE_ELEMENTS) + swath) < unguarded_elements)
			{
				cta.template ProcessFullTile<false>(offset);
			}

			// If the problem is big enough for the last CTA to be in this if-then-block,
			// have it do the remaining guarded work (not first tile)
			if (blockIdx.x == gridDim.x - 1) {
				cta.template ProcessPartialTile<false>(unguarded_elements, work_decomposition.num_elements);
			}

			// Collectively reduce accumulated carry from each thread into output
			// destination (all thread have valid reduction partials)
			cta.OutputToSpine();

		} else {

			// Last CTA does any extra, guarded work (first tile seen)
			cta.template ProcessPartialTile<true>(unguarded_elements, work_decomposition.num_elements);

			// Collectively reduce accumulated carry from each thread into output
			// destination (not every thread may have a valid reduction partial)
			cta.OutputToSpine(work_decomposition.num_elements - unguarded_elements);
		}
	}
};


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
	util::CtaWorkDistribution<typename KernelConfig::SizeT> 	work_decomposition,
	util::CtaWorkProgress											work_progress)
{
	// Shared storage for the kernel
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	UpsweepPass<KernelConfig, KernelConfig::WORK_STEALING>::Invoke(
		d_in,
		d_spine,
		work_decomposition,
		work_progress,
		smem_storage);
}


} // namespace reduction
} // namespace b40c

