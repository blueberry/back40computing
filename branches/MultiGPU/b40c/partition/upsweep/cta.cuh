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
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Abstract upsweep CTA processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>

#include <b40c/partition/upsweep/lanes.cuh>
#include <b40c/partition/upsweep/composites.cuh>
#include <b40c/partition/upsweep/tile.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace partition {
namespace upsweep {



/**
 * CTA
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedCta,							// Derived CTA class
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE,
		typename KernelPolicy> class Tile>			// Derived Tile class to use
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;
	typedef DerivedCta 										Dispatch;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelPolicy::SmemStorage 	&smem_storage;

	// Each thread is responsible for aggregating an unencoded segment of composite counters
	SizeT 								local_counts[KernelPolicy::LANES_PER_WARP][4];

	// Input and output device pointers
	KeyType								*d_in_keys;
	SizeT								*d_spine;

	int 								warp_id;
	int 								warp_idx;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Unrolled tile processing
	 */
	struct UnrollTiles
	{
		// Recurse over counts
		template <int UNROLL_COUNT, int __dummy = 0>
		struct Iterate
		{
			static const int HALF = UNROLL_COUNT / 2;

			template <typename Cta>
			static __device__ __forceinline__ void ProcessTiles(
				Cta *cta, SizeT cta_offset)
			{
				Iterate<HALF>::ProcessTiles(cta, cta_offset);
				Iterate<HALF>::ProcessTiles(cta, cta_offset + (KernelPolicy::TILE_ELEMENTS * HALF));
			}
		};

		// Terminate (process one tile)
		template <int __dummy>
		struct Iterate<1, __dummy>
		{
			template <typename Cta>
			static __device__ __forceinline__ void ProcessTiles(
				Cta *cta, SizeT cta_offset)
			{
				cta->ProcessFullTile(cta_offset);
			}
		};
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			warp_id(threadIdx.x >> B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__)),
			warp_idx(util::LaneId())
	{}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		Dispatch *dispatch = (Dispatch*) this;

		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy> tile;

		// Load keys
		tile.LoadKeys(dispatch, cta_offset);

		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();		// Prevents bucketing from being hoisted up into loads

		// Bucket tile of keys
		tile.Bucket(dispatch);

		// Store keys (if necessary)
		tile.StoreKeys(dispatch, cta_offset);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT cta_out_of_bounds)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Process partial tile if necessary using single loads
		while (cta_offset + threadIdx.x < cta_out_of_bounds) {

			Tile<0, 0, KernelPolicy> tile;

			// Load keys
			tile.LoadKeys(dispatch, cta_offset);

			// Bucket tile of keys
			tile.Bucket(dispatch);

			// Store keys (if necessary)
			tile.StoreKeys(dispatch, cta_offset);

			cta_offset += KernelPolicy::THREADS;
		}
	}


	/**
	 * Processes all tiles
	 */
	__device__ __forceinline__ void ProcessTiles(
		SizeT cta_offset,
		SizeT cta_out_of_bounds)
	{
		Dispatch *dispatch = (Dispatch*) this;

		Composites<KernelPolicy>::ResetCounters(dispatch);
		Lanes<KernelPolicy>::ResetCompositeCounters(dispatch);

		__syncthreads();

		// Unroll batches of full tiles
		const int UNROLLED_ELEMENTS = KernelPolicy::UNROLL_COUNT * KernelPolicy::TILE_ELEMENTS;
		while (cta_offset < cta_out_of_bounds - UNROLLED_ELEMENTS) {

			UnrollTiles::template Iterate<KernelPolicy::UNROLL_COUNT>::ProcessTiles(
				dispatch,
				cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			Composites<KernelPolicy>::ReduceComposites(dispatch);

			__syncthreads();

			// Reset composite counters in lanes
			Lanes<KernelPolicy>::ResetCompositeCounters(dispatch);
		}

		// Unroll single full tiles
		while (cta_offset < cta_out_of_bounds - KernelPolicy::TILE_ELEMENTS) {

			UnrollTiles::template Iterate<1>::ProcessTiles(
				dispatch,
				cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Process partial tile if necessary
		ProcessPartialTile(cta_offset, cta_out_of_bounds);

		__syncthreads();

		// Aggregate back into local_count registers
		Composites<KernelPolicy>::ReduceComposites(dispatch);

		__syncthreads();

		//Final raking reduction of counts by bin, output to spine.

		Composites<KernelPolicy>::PlacePartials(dispatch);

		__syncthreads();

		// Rake-reduce and write out the bin_count reductions
		if (threadIdx.x < KernelPolicy::BINS) {

			SizeT bin_count = util::reduction::SerialReduce<KernelPolicy::AGGREGATED_PARTIALS_PER_ROW>::Invoke(
				smem_storage.aggregate[threadIdx.x]);

			int spine_bin_offset = util::FastMul(gridDim.x, threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					bin_count, d_spine + spine_bin_offset);
		}
	}
};



} // namespace upsweep
} // namespace partition
} // namespace b40c

