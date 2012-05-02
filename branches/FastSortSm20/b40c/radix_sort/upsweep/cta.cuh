/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * CTA-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/partition/upsweep/cta.cuh>

#include <b40c/radix_sort/upsweep/tile.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {

/**
 * Radix sort upsweep reduction CTA
 *
 * Derives from partition::upsweep::Cta
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;

	/**
	 * Shared storage for radix distribution sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			// Composite counter storage
			union {
				char counters[COMPOSITE_LANES][THREADS][4];
				int words[COMPOSITE_LANES][THREADS];
				int direct[COMPOSITE_LANES * THREADS];
			} composite_counters;

			// Final bin reduction storage
			typename TuningPolicy::SizeT aggregate[AGGREGATED_ROWS][PADDED_AGGREGATED_PARTIALS_PER_ROW];
		};
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelPolicy::SmemStorage 	&smem_storage;

	// Shared-memory lanes of composite-counters
	CompostiteCounters<KernelPolicy> 	composite_counters;

	// Thread-local counters for periodically aggregating composite-counter lanes
	AggregateCounters<KernelPolicy>		aggregate_counters;

	// Input and output device pointers
	KeyType								*d_in_keys;
	SizeT								*d_spine;

	int 								warp_id;
	int 								warp_idx;

	char 								*base;


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
	{
		base = (char *) (smem_storage.composite_counters.words[warp_id] + warp_idx);
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		Tile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy> tile;

		// Load keys
		tile.LoadKeys(this, cta_offset);

		// Prevent bucketing from being hoisted (otherwise we don't get the desired outstanding loads)
		if (KernelPolicy::LOADS_PER_TILE > 1) __syncthreads();

		// Bucket tile of keys
		tile.Bucket(this);

		// Store keys (if necessary)
		tile.StoreKeys(this, cta_offset);
	}


	/**
	 * Processes a single load (may have some threads masked off)
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		const SizeT &out_of_bounds)
	{
		// Process partial tile if necessary using single loads
		while (cta_offset + threadIdx.x < out_of_bounds) {

			Tile<0, 0, KernelPolicy> tile;

			// Load keys
			tile.LoadKeys(this, cta_offset);

			// Bucket tile of keys
			tile.Bucket(this);

			// Store keys (if necessary)
			tile.StoreKeys(this, cta_offset);

			cta_offset += KernelPolicy::THREADS;
		}
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		aggregate_counters.ResetCounters(this);
		composite_counters.ResetCompositeCounters(this);


#if 1	// Use deep unrolling for better instruction efficiency

		// Unroll batches of full tiles
		const int UNROLLED_ELEMENTS = KernelPolicy::UNROLL_COUNT * KernelPolicy::TILE_ELEMENTS;
		while (cta_offset  + UNROLLED_ELEMENTS < work_limits.out_of_bounds) {

			UnrollTiles::template Iterate<KernelPolicy::UNROLL_COUNT>::ProcessTiles(
				this,
				cta_offset);
			cta_offset += UNROLLED_ELEMENTS;

			__syncthreads();

			// Aggregate back into local_count registers to prevent overflow
			aggregate_counters.ExtractComposites(this);

			__syncthreads();

			// Reset composite counters in lanes
			composite_counters.ResetCompositeCounters(this);
		}

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset) {

			ProcessFullTile(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;
		}

#else 	// Use shallow unrolling for faster compilation tiles

		// Unroll single full tiles
		while (cta_offset < work_limits.guarded_offset) {

			ProcessFullTile(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			const SizeT UNROLL_MASK = (KernelPolicy::UNROLL_COUNT - 1) << KernelPolicy::LOG_TILE_ELEMENTS;
			if ((cta_offset & UNROLL_MASK) == 0) {

				__syncthreads();

				// Aggregate back into local_count registers to prevent overflow
				aggregate_counters.ExtractComposites(this);

				__syncthreads();

				// Reset composite counters in lanes
				composite_counters.ResetCompositeCounters(this);
			}
		}
#endif

		// Process partial tile if necessary
		ProcessPartialTile(cta_offset, work_limits.out_of_bounds);

		__syncthreads();

		// Aggregate back into local_count registers
		aggregate_counters.ExtractComposites(this);

		__syncthreads();

		//Final raking reduction of counts by bin, output to spine.

		aggregate_counters.ShareCounters(this);

		__syncthreads();

		// Rake-reduce and write out the bin_count reductions
		if (threadIdx.x < KernelPolicy::RADIX_DIGITS) {

			SizeT bin_count = util::reduction::SerialReduce<KernelPolicy::AGGREGATED_PARTIALS_PER_ROW>::Invoke(
				smem_storage.aggregate[threadIdx.x]);

			int spine_bin_offset = util::FastMul(gridDim.x, threadIdx.x) + blockIdx.x;

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					bin_count, d_spine + spine_bin_offset);
		}
	}
};



} // namespace upsweep
} // namespace radix_sort
} // namespace b40c

