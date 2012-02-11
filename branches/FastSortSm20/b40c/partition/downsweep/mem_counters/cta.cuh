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
 * Abstract CTA-processing functionality for partitioning downsweep
 * scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

namespace b40c {
namespace partition {
namespace downsweep {


/**
 * Partitioning downsweep scan CTA
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedCta,									// Derived CTA class
	template <typename Policy> class Tile>			// Derived Tile class to use
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;
	typedef typename KernelPolicy::RakingGrid::LanePartial	LanePartial;

	// Operational details type for short grid
	typedef util::SrtsDetails<typename KernelPolicy::RakingGrid> 		RakingGridDetails;

	typedef DerivedCta Dispatch;

	enum {
		WARP_THREADS 		= B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH),
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelPolicy::SmemStorage 	&smem_storage;

	// Input and output device pointers
	KeyType								*d_in_keys;
	KeyType								*d_out_keys;

	ValueType							*d_in_values;
	ValueType							*d_out_values;

	LanePartial							lane_partial;
	int									*raking_segment;

	int 								*counters;

	SizeT								my_bin_carry;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			lane_partial((LanePartial) (smem_storage.raking_lanes + threadIdx.x + (threadIdx.x >> KernelPolicy::RakingGrid::LOG_PARTIALS_PER_ROW))),
			raking_segment(smem_storage.raking_lanes + (threadIdx.x << KernelPolicy::RakingGrid::LOG_PARTIALS_PER_SEG) + (threadIdx.x >> KernelPolicy::RakingGrid::LOG_SEGS_PER_ROW)),
			counters(&smem_storage.packed_counters_32[0][threadIdx.x])
	{
		// Read bin_carry in parallel
		if (threadIdx.x < KernelPolicy::BINS) {

			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
			my_bin_carry = tex1Dfetch(spine::SpineTex<SizeT>::ref, spine_bin_offset);

			smem_storage.bin_prefixes[threadIdx.x] = 0;
		}

		// Initialize warpscan identity regions
		if ((KernelPolicy::THREADS == KernelPolicy::RAKING_THREADS) || (threadIdx.x < KernelPolicy::RAKING_THREADS)) {

		int tid = threadIdx.x + (threadIdx.x & (~31));
			smem_storage.warpscan[tid] = 0;
		}
	}


	/**
	 * Process tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<KernelPolicy> tile;

		tile.Partition(
			cta_offset,
			guarded_elements,
			(Dispatch *) this);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT pack_offset = smem_storage.packed_offset;

		// Process full tiles of tile_elements
		while (pack_offset < smem_storage.packed_offset_limit) {

			ProcessTile(pack_offset);
			pack_offset += (KernelPolicy::TILE_ELEMENTS / KernelPolicy::PACK_SIZE);
		}

		// Clean up last partial tile with guarded-io
		if (work_limits.guarded_elements) {

			ProcessTile(
				pack_offset,
				work_limits.guarded_elements);
		}
	}
};


} // namespace downsweep
} // namespace partition
} // namespace b40c

