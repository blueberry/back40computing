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
		T (*)[KernelPolicy::LOAD_VEC_SIZE],
		Flag (*)[KernelPolicy::LOAD_VEC_SIZE]> 		DataSoa;

	// This kernel can only operate in inclusive scan mode if the it's the final kernel
	// in the scan pass
	static const bool KERNEL_EXCLUSIVE = (!KernelPolicy::FINAL_KERNEL || KernelPolicy::EXCLUSIVE);

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 	srts_soa_details;

	// The tuple value we will accumulate (in raking threads only)
	SoaTuple 		carry;

	// Input device pointers
	T 				*d_partials_in;
	Flag 			*d_flags_in;

	// Output device pointer
	T 				*d_partials_out;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
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

		T				partials[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		Flag			flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];
		int 			is_head[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];


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
			 * RepairHeads
			 */
			static __device__ __forceinline__ void RepairHeads(Tile *tile)
			{
				// Set the partials of flagged items to identity
				if (tile->is_head[LOAD][VEC]) {
					tile->partials[LOAD][VEC] = 0;
				}

				// Next
				Iterate<LOAD, VEC + 1>::RepairHeads(tile);
			}
		};

		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// RepairHeads
			static __device__ __forceinline__ void RepairHeads(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::RepairHeads(tile);
			}
		};

		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// RepairHeads
			static __device__ __forceinline__ void RepairHeads(Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Performs any cleanup work
		 */
		__device__ __forceinline__ void RepairHeads()
		{
			if (KernelPolicy::FINAL_KERNEL && KernelPolicy::EXCLUSIVE) {
				Iterate<0, 0>::RepairHeads(this);
			}
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
		T 				*d_partials_in,
		Flag 			*d_flags_in,
		T 				*d_partials_out,
		T 				spine_partial = KernelPolicy::Identity()) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.partials_raking_elements,
					smem_storage.flags_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.flags_warpscan),
				KernelPolicy::SoaTupleIdentity()),
			d_partials_in(d_partials_in),
			d_flags_in(d_flags_in),
			d_partials_out(d_partials_out),
			carry(												// Seed carry with spine partial & flag identity
				spine_partial,
				0)
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

