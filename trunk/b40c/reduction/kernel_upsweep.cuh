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

#include <b40c/util/srts_details.cuh>
#include <b40c/util/work_distribution.cuh>
#include <b40c/util/work_progress.cuh>
#include <b40c/reduction/reduction_cta.cuh>

namespace b40c {
namespace reduction {



/**
 * Upsweep reduction pass (non-workstealing)
 */
template <typename KernelConfig, bool WORK_STEALING>
struct UpsweepReductionPass
{
	static __device__ __forceinline__ void Invoke(
		typename KernelConfig::T 									*&d_in,
		typename KernelConfig::T 									*&d_out,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> 	&work_decomposition,
		const util::WorkProgress 									&work_progress)
	{
		typedef typename KernelConfig::SrtsDetails SrtsDetails;
		typedef ReductionCta<KernelConfig> ReductionCta;
		typedef typename ReductionCta::T T;
		typedef typename ReductionCta::SizeT SizeT;

		// Shared storage for CTA processing
		__shared__ uint4 smem_pool[KernelConfig::SRTS_GRID_QUADS];
		__shared__ T warpscan[2][B40C_WARP_THREADS(KernelConfig::CUDA_ARCH)];

		// SRTS grid details
		SrtsDetails srts_detail(smem_pool, warpscan);

		// CTA processing abstraction
		ReductionCta cta(srts_detail, d_in, d_out);

		// Determine our threadblock's work range
		SizeT cta_offset;			// Offset at which this CTA begins processing
		SizeT cta_elements;			// Total number of elements for this CTA to process
		SizeT guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
		SizeT guarded_elements;		// Number of elements in partially-full tile

		work_decomposition.GetCtaWorkLimits<ReductionCta::LOG_TILE_ELEMENTS, ReductionCta::LOG_SCHEDULE_GRANULARITY>(
			cta_offset, cta_elements, guarded_offset, guarded_elements);

		SizeT out_of_bounds = cta_offset + cta_elements;

		// Process full tiles of tile_elements
		while (cta_offset < guarded_offset) {

			cta.ProcessTile<true>(cta_offset, out_of_bounds);
			cta_offset += ReductionCta::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (guarded_elements) {
			cta.ProcessTile<false>(cta_offset, out_of_bounds);
		}

		// Collectively reduce accumulated carry from each thread into output destination
		cta.FinalReduction();
	}
};


/**
 * Upsweep reduction pass (workstealing)
 */
template <typename KernelConfig>
struct UpsweepReductionPass <KernelConfig, true>
{
	static __device__ __forceinline__ void Invoke(
		typename KernelConfig::T 									*&d_in,
		typename KernelConfig::T 									*&d_out,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> 	&work_decomposition,
		const util::WorkProgress 									&work_progress)
	{
		typedef typename KernelConfig::SrtsDetails SrtsDetails;
		typedef ReductionCta<KernelConfig> ReductionCta;
		typedef typename ReductionCta::T T;
		typedef typename ReductionCta::SizeT SizeT;

		// Shared storage for CTA processing
		__shared__ uint4 smem_pool[KernelConfig::SRTS_GRID_QUADS];
		__shared__ T warpscan[2][B40C_WARP_THREADS(KernelConfig::CUDA_ARCH)];

		// SRTS grid details
		SrtsDetails srts_detail(smem_pool, warpscan);

		// CTA processing abstraction
		ReductionCta cta(srts_detail, d_in, d_out);

		// The offset at which this CTA performs tile processing
		__shared__ SizeT cta_offset;

		// First CTA resets the work progress for the next pass
		if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
			work_progress.PrepareNext();
		}

		// Steal full-tiles of work, incrementing progress counter
		SizeT unguarded_elements = work_decomposition.num_elements & (~(ReductionCta::TILE_ELEMENTS - 1));
		while (true) {

			// Thread zero atomically steals work from the progress counter
			if (threadIdx.x == 0) {
				cta_offset = work_progress.Steal<ReductionCta::TILE_ELEMENTS>();
			}

			__syncthreads();		// Protect cta_offset

			if (cta_offset >= unguarded_elements) {
				// All done
				break;
			}

			cta.ProcessTile<true>(cta_offset, unguarded_elements);
		}

		// Last CTA does any extra, guarded work
		if (blockIdx.x == gridDim.x - 1) {
			cta.ProcessTile<false>(unguarded_elements, work_decomposition.num_elements);
		}

		// Collectively reduce accumulated carry from each thread into output destination
		cta.FinalReduction();
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
void UpsweepReductionKernel(
	typename KernelConfig::T 			*d_in,
	typename KernelConfig::T 			*d_spine,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition,
	util::WorkProgress					work_progress)
{
	typename KernelConfig::T *d_spine_partial = d_spine + blockIdx.x;

	UpsweepReductionPass<KernelConfig, KernelConfig::WORK_STEALING>::Invoke(
		d_in,
		d_spine_partial,
		work_decomposition,
		work_progress);
}


} // namespace reduction
} // namespace b40c

