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
 * CTA-processing functionality for consecutive removal downsweep
 * scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/operators.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace consecutive_removal {
namespace downsweep {


/**
 * Consecutive removal downsweep scan CTA
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 				T;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::SpineType		SpineType;
	typedef typename KernelPolicy::LocalFlag		LocalFlag;			// Type for noting local discontinuities
	typedef typename KernelPolicy::SrtsType			SrtsType;			// Type for local SRTS prefix sum
	typedef typename KernelPolicy::SrtsDetails 		SrtsDetails;
	typedef typename KernelPolicy::SmemStorage 		SmemStorage;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running accumulator for the number of discontinuities observed by
	// the CTA over its tile-processing lifetime (managed in each raking thread)
	SpineType		carry;

	// Input and output device pointers
	T 				*d_in;
	T 				*d_out;
	SizeT 			*d_num_compacted;

	// Pool of storage for compacting a tile of values
	T 				*exchange;

	// Operational details for SRTS scan grid
	SrtsDetails 	srts_details;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Two-phase scatter tile specialization
	 */
	template <bool FIRST_TILE, bool TWO_PHASE_SCATTER = KernelPolicy::TWO_PHASE_SCATTER>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];					// Tile of elements
		T compacted_data[KernelPolicy::TILE_ELEMENTS_PER_THREAD][1];						// Tile of compacted elements
		LocalFlag flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of discontinuity flags
		SrtsType ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of local scatter offsets

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
			// Load data tile, initializing discontinuity flags
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::template LoadDiscontinuity<FIRST_TILE>(
					data,
					flags,
					cta->d_in + cta_offset,
					guarded_elements);

			// Copy discontinuity flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, flags);

			// Scan tile of ranks
			SrtsType unique_elements = util::scan::CooperativeTileScan<
				SrtsDetails,
				KernelPolicy::LOAD_VEC_SIZE,
				true,							// exclusive
				util::Operators<SrtsType>::Sum>::ScanTile(
					cta->srts_details, ranks);

			// Barrier sync to protect smem exchange storage
			__syncthreads();

			// Scatter valid data into smem exchange, predicated on flags (treat
			// data, flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				util::io::st::NONE>::Scatter(
					cta->exchange,
					(T*) data,
					(LocalFlag*) flags,
					(SrtsType*) ranks);

			// Barrier sync to protect smem exchange storage
			__syncthreads();

			// Gather compacted data from smem exchange (in 1-element stride loads)
			util::io::LoadTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0, 											// Vec-1
				KernelPolicy::THREADS,
				util::io::ld::NONE>::LoadValid(
					compacted_data,
					cta->exchange,
					unique_elements);

			// Scatter compacted data to global output
			util::io::StoreTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0, 											// Vec-1
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Store(
					compacted_data,
					cta->d_out + cta->carry,
					unique_elements);

			// Update running discontinuity count for CTA
			cta->carry += unique_elements;

			// Barrier sync to protect smem exchange storage
			__syncthreads();
		}
	};


	/**
	 * Direct-scatter tile specialization
	 */
	template <bool FIRST_TILE>
	struct Tile<FIRST_TILE, false>
	{
		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];					// Tile of elements
		LocalFlag flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of discontinuity flags
		SrtsType ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];			// Tile of global scatter offsets


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
			// Load data tile, initializing discontinuity flags
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::template LoadDiscontinuity<FIRST_TILE>(
					data,
					flags,
					cta->d_in + cta_offset,
					guarded_elements);

			// Copy discontinuity flags into ranks
			util::io::InitializeTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, flags);

			// Scan tile of ranks, seed with carry (maintain carry in raking threads)
			util::scan::CooperativeTileScan<
				SrtsDetails,
				KernelPolicy::LOAD_VEC_SIZE,
				true,							// exclusive
				util::Operators<SrtsType>::Sum>::ScanTileWithCarry(
					cta->srts_details,
					ranks,
					cta->carry);

			// Scatter valid vertex_id into smem exchange, predicated on flags (treat
			// vertex_id, flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					cta->d_out,
					(T*) data,
					(LocalFlag*) flags,
					(SrtsType*) ranks);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage &smem_storage,
		T 			*d_in,
		T 			*d_out,
		SizeT 		*d_num_compacted,
		SpineType	spine_partial = 0) :

			srts_details(
				smem_storage.raking_elements,
				smem_storage.warpscan,
				0),
			exchange(smem_storage.exchange),
			d_in(d_in),
			d_out(d_out),
			d_num_compacted(d_num_compacted),
			carry(spine_partial) 			// Seed carry with spine partial
	{}


	/**
	 * Process a single tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
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

		// Output number of compacted items
		if (work_limits.last_block && (threadIdx.x == 0)) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_num_compacted);
		}
	}

};


} // namespace downsweep
} // namespace consecutive_removal
} // namespace b40c

