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
	bool _FLOP_TURN,
	typename DerivedCta,									// Derived CTA class
	template <typename Policy> class Tile>					// Derived Tile class to use
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;

	typedef DerivedCta Dispatch;

	enum {
		WARP_THREADS 				= B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH),
		FLOP_TURN					= _FLOP_TURN,

		LOG_MEM_BANKS				= B40C_LOG_MEM_BANKS(KernelPolicy::CUDA_ARCH),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		BANK_PADDING 				= 1,				// Whether or not to insert padding for exchanging keys
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelPolicy::SmemStorage 	&smem_storage;

	KeyType								*d_keys0;
	KeyType								*d_keys1;

	ValueType							*d_values0;
	ValueType							*d_values1;

	int									*raking_segment;
	unsigned short						*bin_counter;

	SizeT								my_bin_carry;

	int 								warp_id;
	volatile int 						*warpscan;

	KeyType 							*base_gather_offset;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys0,
		KeyType 		*d_keys1,
		ValueType 		*d_values0,
		ValueType 		*d_values1,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_keys0(d_keys0),
			d_keys1(d_keys1),
			d_values0(d_values0),
			d_values1(d_values1),
			raking_segment(smem_storage.raking_grid[threadIdx.x]),
			base_gather_offset(smem_storage.key_exchange + threadIdx.x + ((BANK_PADDING) ? (threadIdx.x >> LOG_MEM_BANKS) : 0))
	{
		int counter_lane = threadIdx.x & (KernelPolicy::SCAN_LANES - 1);
		int sub_counter = threadIdx.x >> (KernelPolicy::LOG_BINS - 1);
		bin_counter = &smem_storage.packed_counters[counter_lane][0][sub_counter];

		warp_id = (threadIdx.x & (~31));
		warpscan = smem_storage.warpscan + 32 + threadIdx.x + warp_id;

		if ((KernelPolicy::THREADS == KernelPolicy::BINS) || (threadIdx.x < KernelPolicy::BINS)) {

			// Read bin_carry in parallel
			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
			my_bin_carry = tex1Dfetch(spine::SpineTex<SizeT>::ref, spine_bin_offset);
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

