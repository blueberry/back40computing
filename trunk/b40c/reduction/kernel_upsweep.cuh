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

#include <b40c/util/work_distribution.cuh>
#include <b40c/reduction/kernel_cta.cuh>

namespace b40c {
namespace reduction {



/**
 * Upsweep reduction pass (non-workstealing)
 */
template <typename ReductionKernelConfig, bool WORK_STEALING>
struct UpsweepReductionPass
{
	static __device__ __forceinline__ void Invoke(
		typename ReductionKernelConfig::T 			*&d_in,
		typename ReductionKernelConfig::T 			*&d_out,
		typename ReductionKernelConfig::SizeT 		* __restrict &d_work_progress,
		util::CtaWorkDistribution<typename ReductionKernelConfig::SizeT> &work_decomposition,
		const int &progress_selector)
	{
		typedef ReductionCta<ReductionKernelConfig> ReductionCta;
		typedef typename ReductionCta::T T;
		typedef typename ReductionCta::SizeT SizeT;

		ReductionCta cta(d_in, d_out);

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

		// Collectively reduce accumulated carry from each thread
		cta.CollectiveReduction();
	}
};



/**
 * Upsweep reduction pass (workstealing)
 */
template <typename ReductionKernelConfig>
struct UpsweepReductionPass <ReductionKernelConfig, true>
{
	static __device__ __forceinline__ void Invoke(
		typename ReductionKernelConfig::T 			*&d_in,
		typename ReductionKernelConfig::T 			*&d_out,
		typename ReductionKernelConfig::SizeT 		* __restrict &d_work_progress,
		const util::CtaWorkDistribution<typename ReductionKernelConfig::SizeT> &work_decomposition,
		const int &progress_selector)
	{
		typedef ReductionCta<ReductionKernelConfig> ReductionCta;
		typedef typename ReductionCta::T T;
		typedef typename ReductionCta::SizeT SizeT;

		ReductionCta cta(d_in, d_out);

		// The offset at which this CTA performs tile processing
		__shared__ SizeT cta_offset;

		// First CTA resets the work progress for the next pass
		if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
			d_work_progress[progress_selector ^ 1] = 0;
		}

		// Steal full-tiles of work, incrementing progress counter
		SizeT unguarded_elements = work_decomposition.num_elements & (~(ReductionCta::TILE_ELEMENTS - 1));
		while (true) {

			// Thread zero atomically steals work from the progress counter
			if (threadIdx.x == 0) {
				cta_offset = atomicAdd(&d_work_progress[progress_selector], ReductionCta::TILE_ELEMENTS);
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

		// Collectively reduce accumulated carry from each thread
		cta.CollectiveReduction();
	}
};


/******************************************************************************
 * Upsweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Upsweep reduction kernel entry point
 */
template <typename ReductionKernelConfig>
__launch_bounds__ (ReductionKernelConfig::THREADS, ReductionKernelConfig::CTA_OCCUPANCY)
__global__
void UpsweepReductionKernel(
	typename ReductionKernelConfig::T 			*d_in,
	typename ReductionKernelConfig::T 			*d_spine,
	typename ReductionKernelConfig::SizeT 		* __restrict d_work_progress,
	util::CtaWorkDistribution<typename ReductionKernelConfig::SizeT> work_decomposition,
	int progress_selector)
{
	typename ReductionKernelConfig::T *d_spine_partial = d_spine + blockIdx.x;

	UpsweepReductionPass<ReductionKernelConfig, ReductionKernelConfig::WORK_STEALING>::Invoke(
		d_in,
		d_spine_partial,
		d_work_progress,
		work_decomposition,
		progress_selector);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename ReductionKernelConfig>
void __wrapper__device_stub_UpsweepReductionKernel(
	typename ReductionKernelConfig::T 			*&,
	typename ReductionKernelConfig::T 			*&,
	typename ReductionKernelConfig::SizeT 		* __restrict &,
	util::CtaWorkDistribution<typename ReductionKernelConfig::SizeT> &,
	int &) {}



} // namespace reduction
} // namespace b40c

