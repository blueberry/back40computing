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
 * CTA-processing functionality for consecutive reduction downsweep
 * scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace consecutive_reduction {
namespace downsweep {


/**
 * Consecutive reduction downsweep scan CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 				KeyType;
	typedef typename KernelPolicy::ValueType			ValueType;
	typedef typename KernelPolicy::SizeT 				SizeT;

	typedef typename KernelPolicy::SpinePartialType		SpinePartialType;
	typedef typename KernelPolicy::SpineFlagType		SpineFlagType;
	typedef typename KernelPolicy::SpineSoaTuple 		SpineSoaTuple;

	typedef typename KernelPolicy::LocalFlag			LocalFlag;			// Type for noting local discontinuities
	typedef typename KernelPolicy::RankType				RankType;			// Type for local SRTS prefix sum

	typedef typename KernelPolicy::SrtsSoaDetails 		SrtsSoaDetails;
	typedef typename KernelPolicy::SrtsSoaTuple 		SrtsSoaTuple;

	typedef util::Tuple<
		ValueType (*)[KernelPolicy::LOAD_VEC_SIZE],
		RankType (*)[KernelPolicy::LOAD_VEC_SIZE]> 		DataSoa;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 	srts_soa_details;

	// The spine value-flag tuple value we will accumulate (in raking threads only)
	SpineSoaTuple 	carry;

	// Device pointers
	KeyType 		*d_in_keys;
	KeyType			*d_out_keys;
	ValueType 		*d_in_values;
	ValueType 		*d_out_values;
	SizeT			*d_num_compacted;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile (direct-scatter specialization)
	 */
	template <
		bool FIRST_TILE,
		bool TWO_PHASE_SCATTER = KernelPolicy::TWO_PHASE_SCATTER>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE = KernelPolicy::LOADS_PER_TILE,
			LOAD_VEC_SIZE = KernelPolicy::LOAD_VEC_SIZE,
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		KeyType			keys[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		ValueType		values[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		LocalFlag		head_flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Tile of discontinuity flags
		RankType 		ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of global scatter offsets


		//---------------------------------------------------------------------
		// Iteration Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * DecrementRanks
			 */
			static __device__ __forceinline__ void DecrementRanks(Tile *tile)
			{
				tile->ranks[LOAD][VEC]--;

				// Next
				Iterate<LOAD, VEC + 1>::DecrementRanks(tile);
			}
		};

		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// DecrementRanks
			static __device__ __forceinline__ void DecrementRanks(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::DecrementRanks(tile);
			}
		};

		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// DecrementRanks
			static __device__ __forceinline__ void DecrementRanks(Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Process tile
		 */
		template <typename Cta>
		__device__ __forceinline__ void Process(
			SizeT cta_offset,
			const SizeT &guarded_elements,
			Cta *cta)
		{
			// Load keys, initializing discontinuity head_flags
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::template LoadDiscontinuity<FIRST_TILE>(
					keys,
					head_flags,
					cta->d_in_keys + cta_offset,
					guarded_elements);

			// Load values
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::LoadValid(
					values,
					cta->d_in_values + cta_offset,
					guarded_elements);

			// Copy discontinuity head_flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, head_flags);

			// SOA-scan tile of tuple pairs
			util::scan::soa::CooperativeSoaTileScan<
				SrtsSoaDetails,
				KernelPolicy::LOAD_VEC_SIZE,
				true,								// Exclusive scan
				KernelPolicy::SoaScanOp>::template ScanTileWithCarry<DataSoa>(
					cta->srts_soa_details,
					DataSoa(values, ranks),
					cta->carry);					// Seed with carry, maintain carry in raking threads

			// Scatter valid keys directly to global output, predicated on head_flags
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out_keys,
					(KeyType*) keys,				// Treat as linear arrays
					(LocalFlag*) head_flags,
					(RankType*) ranks);

			// Decrement scatter ranks for values
			Iterate<0, 0>::DecrementRanks(this);

			// First CTA unsets the first head flag of first tile
			if (FIRST_TILE && (blockIdx.x == 0) && (threadIdx.x == 0)) {
				head_flags[0][0] = 0;
			}

			// Scatter valid reduced values directly to global output, predicated on head_flags
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out_values,
					(ValueType*) values,			// Treat as linear arrays
					(LocalFlag*) head_flags,
					(RankType*) ranks);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SizeT			*d_num_compacted,
		SpineSoaTuple	spine_partial = KernelPolicy::SpineTupleIdentity()) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.ranks_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.ranks_warpscan),
				KernelPolicy::SrtsTupleIdentity()),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			d_num_compacted(d_num_compacted),
			carry(spine_partial)
	{}


	/**
	 * Process a single tile
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<FIRST_TILE> tile;

		tile.Process(cta_offset, guarded_elements, this);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		if (cta_offset < work_limits.guarded_offset) {

			// Process at least one full tile of tile_elements (first tile)
			ProcessTile<true>(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;

			while (cta_offset < work_limits.guarded_offset) {
				// Process more full tiles (not first tile)
				ProcessTile<false>(cta_offset);
				cta_offset += KernelPolicy::TILE_ELEMENTS;
			}

			// Clean up last partial tile with guarded-io (not first tile)
			if (work_limits.guarded_elements) {
				ProcessTile<false>(
					cta_offset,
					work_limits.guarded_elements);
			}

		} else {

			// Clean up last partial tile with guarded-io (first tile)
			ProcessTile<true>(
				cta_offset,
				work_limits.guarded_elements);
		}

		// The last block writes final values
		if ((work_limits.last_block) &&
			(threadIdx.x == SrtsSoaDetails::CUMULATIVE_THREAD) &&
			(d_num_compacted != NULL))
		{
			// Output the final reduced value
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry.t0, d_out_values + carry.t1 - 1);

			// Output the number of compacted items
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry.t1, d_num_compacted);
		}
	}
};


} // namespace downsweep
} // namespace consecutive_reduction
} // namespace b40c

