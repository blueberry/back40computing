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
#include <b40c/scan/kernel_tile.cuh>

namespace b40c {
namespace scan {



/**
 * Downsweep scan pass
 */
template <typename ScanKernelConfig>
__device__ __forceinline__ void DownsweepScanPass(
		typename ScanKernelConfig::T 			* &d_in,
		typename ScanKernelConfig::T 			* &d_out,
		typename ScanKernelConfig::T 			* __restrict &d_spine_partial,
		util::CtaWorkDistribution<typename ScanKernelConfig::SizeT> &work_decomposition)
	{
		typedef ScanTile<ScanKernelConfig> Tile;
		typedef typename Tile::T T;
		typedef typename Tile::SizeT SizeT;

		// Shared memory pool
		__shared__ uint4 smem_pool[Tile::SMEM_QUADS];

		// Warpscan shared memory
		__shared__ ScanType warpscan[2][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

		// The carry that thread-0 will aggregate
		T carry = Tile::Identity();		// The value we will accumulate

		// Determine our threadblock's work range
		SizeT cta_offset;			// Offset at which this CTA begins processing
		SizeT cta_elements;			// Total number of elements for this CTA to process
		SizeT guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
		SizeT guarded_elements;		// Number of elements in partially-full tile

		work_decomposition.GetCtaWorkLimits<Tile::LOG_TILE_ELEMENTS, Tile::LOG_SCHEDULE_GRANULARITY>(
			cta_offset, cta_elements, guarded_offset, guarded_elements);

		SizeT out_of_bounds = cta_offset + cta_elements;

		T *primary_grid = reinterpret_cast<ScanType*>(smem_pool);
		T *primary_base_partial = Config::PrimaryGrid::BasePartial(primary_grid);
		T *primary_raking_seg = NULL;
		T *secondary_base_partial = NULL;
		T *secondary_raking_seg = NULL;

		ScanType carry = 0;

		// Initialize partial-placement and raking offset pointers
		if (threadIdx.x < Config::PrimaryGrid::RAKING_THREADS) {

			primary_raking_seg = Config::PrimaryGrid::RakingSegment(primary_grid);

			ScanType *secondary_grid = reinterpret_cast<ScanType*>(smem_pool + Config::PrimaryGrid::SMEM_BYTES);		// Offset by the primary grid
			secondary_base_partial = Config::SecondaryGrid::BasePartial(secondary_grid);
			if (Config::TWO_LEVEL_GRID && (threadIdx.x < Config::SecondaryGrid::RAKING_THREADS)) {
				secondary_raking_seg = Config::SecondaryGrid::RakingSegment(secondary_grid);
			}

			// Initialize warpscan
			if (threadIdx.x < B40C_WARP_THREADS(__B40C_CUDA_ARCH__)) {
				warpscan[0][threadIdx.x] = 0;
			}
		}




		// Process full tiles of tile_elements
		while (cta_offset < guarded_offset) {

			Tile::ProcessTile<true>(d_in, cta_offset, out_of_bounds, carry);
			cta_offset += Tile::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (guarded_elements) {
			Tile::ProcessTile<false>(d_in, cta_offset, out_of_bounds, carry);
		}

		// Collectively scan accumulated carry from each thread
		Tile::CollectiveScan(carry, d_out);
	}
};



/******************************************************************************
 * Downsweep Scan Kernel Entrypoint
 ******************************************************************************/

/**
 * Downsweep scan kernel entry point
 */
template <typename ScanKernelConfig>
__launch_bounds__ (ScanKernelConfig::THREADS, ScanKernelConfig::CTA_OCCUPANCY)
__global__
void DownsweepScanKernel(
	typename ScanKernelConfig::T 			* d_in,
	typename ScanKernelConfig::T 			* d_out,
	typename ScanKernelConfig::T 			* __restrict d_spine,
	util::CtaWorkDistribution<typename ScanKernelConfig::SizeT> work_decomposition)
{
	typename ScanKernelConfig::T *d_spine_partial = d_spine + blockIdx.x;

	DownsweepScanPass(d_in, d_out, d_spine_partial, work_decomposition);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename ScanKernelConfig>
void __wrapper__device_stub_DownsweepScanKernel(
	typename ScanKernelConfig::T 			* &,
	typename ScanKernelConfig::T 			* &,
	typename ScanKernelConfig::T 			* __restrict &,
	util::CtaWorkDistribution<typename ScanKernelConfig::SizeT> &) {}



} // namespace scan
} // namespace b40c

