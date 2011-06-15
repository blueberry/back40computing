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

	typedef typename KernelPolicy::KeyType 			KeyType;
	typedef typename KernelPolicy::ValueType		ValueType;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::SpineType		SpineType;
	typedef typename KernelPolicy::LocalFlag		LocalFlag;			// Type for noting local discontinuities
	typedef typename KernelPolicy::RankType			RankType;			// Type for local SRTS prefix sum

	typedef typename KernelPolicy::SrtsSoaDetails 	SrtsSoaDetails;
	typedef typename KernelPolicy::SoaTuple 		SoaTuple;

	typedef util::Tuple<
		ValueType (*)[KernelPolicy::LOAD_VEC_SIZE],
		RankType (*)[KernelPolicy::LOAD_VEC_SIZE]> 		DataSoa;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 	srts_soa_details;

	// The tuple value we will accumulate (in raking threads only)
	SoaTuple 		carry;

	// Device pointers
	KeyType 		*d_in_keys;
	KeyType			*d_out_keys;
	ValueType 		*d_in_values;
	ValueType 		*d_out_values;


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
		LocalFlag		head_flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		RankType 		ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of global scatter offsets


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
				KernelPolicy::READ_MODIFIER>::template LoadDiscontinuity<FIRST_TILE, false>(	// Do not flag first element of first tile of first cta
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

			// Scan tile, seed with carry (maintain carry in raking threads)
			util::scan::soa::CooperativeSoaTileScan<
				SrtsSoaDetails,
				KernelPolicy::LOAD_VEC_SIZE,
				true,																// exclusive
				KernelPolicy::SoaScanOp>::template ScanTileWithCarry<DataSoa>(
					srts_soa_details,
					DataSoa(values, ranks),
					carry);

			// Scatter valid vertex_id into smem exchange, predicated on head_flags (treat
			// vertex_id, head_flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out,
					(T*) data,
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
		KeyType 		*d_in_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		SoaTuple		spine_partial = KernelPolicy::SoaTupleIdentity()) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.ranks_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.ranks_warpscan),
				KernelPolicy::SoaTupleIdentity()),
			d_in_keys(d_in_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			carry(spine_partial)
	{}


	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tiles of consecutive reduction elements and flags
		Tile tile;

		// Load tile of partials
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				tile.partials,
				d_partials_in + cta_offset,
				guarded_elements);

		// Load tile of flags
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				tile.flags,
				d_flags_in + cta_offset,
				guarded_elements);

		// Copy head flags (since we will trash them during scan)
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(tile.is_head, tile.flags);

		// Scan tile with carry update in raking threads
		util::scan::soa::CooperativeSoaTileScan<
			SrtsSoaDetails,
			KernelPolicy::LOAD_VEC_SIZE,
			KERNEL_EXCLUSIVE,
			KernelPolicy::SoaScanOp>::template ScanTileWithCarry<DataSoa>(
				srts_soa_details,
				DataSoa(tile.partials, tile.flags),
				carry);

		// Fix up segment heads if exclusive scan
		tile.RepairHeads();

		// Store tile of partials
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Store(
				tile.partials, d_partials_out + cta_offset, guarded_elements);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT cta_offset = work_limits.offset;

		// Process full tiles of tile_elements
		while (cta_offset < work_limits.guarded_offset) {

			ProcessTile(cta_offset);
			cta_offset += KernelPolicy::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-io
		if (work_limits.guarded_elements) {
			ProcessTile(
				cta_offset,
				work_limits.guarded_elements);
		}
	}
};


} // namespace downsweep
} // namespace consecutive_reduction
} // namespace b40c

