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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Distribution sort downsweep scan-scatter kernel.
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/device_intrinsics.cuh>

#include <b40c/radix_sort/distribution/downsweep/cta.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace downsweep {


/**
 * Downsweep scan-scatter pass
 */
template <typename KernelConfig, typename SmemStorage>
__device__ __forceinline__ void DownsweepPass(
	int 								*&d_selectors,
	typename KernelConfig::SizeT 		*&d_spine,
	typename KernelConfig::KeyType 		*&d_keys0,
	typename KernelConfig::KeyType 		*&d_keys1,
	typename KernelConfig::ValueType 	*&d_values0,
	typename KernelConfig::ValueType 	*&d_values1,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
	SmemStorage	&smem_storage)
{
	typedef typename KernelConfig::KeyType 				KeyType;
	typedef typename KernelConfig::SizeT 				SizeT;
	typedef Cta<KernelConfig, SmemStorage> 				Cta;
	typedef typename KernelConfig::Grid::LanePartial	LanePartial;

	LanePartial base_partial = KernelConfig::Grid::MyLanePartial(smem_storage.smem_pool.raking_lanes);
	int *raking_segment = 0;

	if (threadIdx.x < KernelConfig::Grid::RAKING_THREADS) {

		// initalize lane warpscans
		int warpscan_lane = threadIdx.x >> KernelConfig::Grid::LOG_RAKING_THREADS_PER_LANE;
		int warpscan_tid = threadIdx.x & (KernelConfig::Grid::RAKING_THREADS_PER_LANE - 1);
		smem_storage.lanes_warpscan[warpscan_lane][0][warpscan_tid] = 0;

		raking_segment = KernelConfig::Grid::MyRakingSegment(smem_storage.smem_pool.raking_lanes);

		// initialize digit warpscans
		if (threadIdx.x < KernelConfig::RADIX_DIGITS) {

			// Initialize digit_warpscan
			smem_storage.digit_warpscan[0][threadIdx.x] = 0;

			// Determine where to read our input
			if (KernelConfig::EARLY_EXIT) {

				// We have early-exit-upon-homogeneous-digits enabled

				const int SELECTOR_IDX 			= (KernelConfig::CURRENT_PASS) & 0x1;
				const int NEXT_SELECTOR_IDX 	= (KernelConfig::CURRENT_PASS + 1) & 0x1;

				smem_storage.selector = (KernelConfig::CURRENT_PASS == 0) ? 0 : d_selectors[SELECTOR_IDX];

				// Determine whether or not we have work to do and setup the next round
				// accordingly.  We can do this by looking at the first-block's
				// histograms and counting the number of digits with counts that are
				// non-zero and not-the-problem-size.
				if (KernelConfig::PreprocessTraits::MustApply || KernelConfig::PostprocessTraits::MustApply) {
					smem_storage.non_trivial_digit_pass = true;
				} else {
					int first_block_carry = d_spine[util::FastMul(gridDim.x, threadIdx.x)];
					int predicate = ((first_block_carry > 0) && (first_block_carry < work_decomposition.num_elements));
					smem_storage.non_trivial_digit_pass = util::TallyWarpVote<KernelConfig::RADIX_BITS>(
						predicate,
						smem_storage.smem_pool.raking_lanes);
				}

				// Let the next round know which set of buffers to use
				if (blockIdx.x == 0) {
					d_selectors[NEXT_SELECTOR_IDX] = smem_storage.selector ^ smem_storage.non_trivial_digit_pass;
				}
			}

			// Determine our threadblock's work range
			work_decomposition.template GetCtaWorkLimits<
				KernelConfig::LOG_TILE_ELEMENTS,
				KernelConfig::LOG_SCHEDULE_GRANULARITY>(smem_storage.work_limits);
		}
	}

	// Sync to acquire non_trivial_digit_pass, selector, and work limits
	__syncthreads();

	// Short-circuit this entire cycle
	if (KernelConfig::EARLY_EXIT && !smem_storage.non_trivial_digit_pass) return;

	if ((KernelConfig::EARLY_EXIT && smem_storage.selector) || (!KernelConfig::EARLY_EXIT && (KernelConfig::CURRENT_PASS & 0x1))) {

		// d_keys1 -> d_keys0
		Cta cta(
			smem_storage,
			d_keys1,
			d_keys0,
			d_values1,
			d_values0,
			d_spine,
			base_partial,
			raking_segment);

		cta.ProcessTiles();

	} else {

		// d_keys0 -> d_keys1
		Cta cta(
			smem_storage,
			d_keys0,
			d_keys1,
			d_values0,
			d_values1,
			d_spine,
			base_partial,
			raking_segment);

		cta.ProcessTiles();
	}
}


/**
 * Downsweep scan-scatter kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void DownsweepKernel(
	int 								*d_selectors,
	typename KernelConfig::SizeT 		*d_spine,
	typename KernelConfig::KeyType 		*d_keys0,
	typename KernelConfig::KeyType 		*d_keys1,
	typename KernelConfig::ValueType 	*d_values0,
	typename KernelConfig::ValueType 	*d_values1,
	util::CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition)
{
	// Shared memory pool
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	DownsweepPass<KernelConfig>(
		d_selectors,
		d_spine,
		d_keys0,
		d_keys1,
		d_values0,
		d_values1,
		work_decomposition,
		smem_storage);
}



} // namespace downsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

