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
 * Tile-processing functionality for consecutive-removal downsweep
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


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig>
struct DownsweepCta : KernelConfig
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::T 				T;					// Input type for detecting consecutive discontinuities
	typedef typename KernelConfig::FlagCount 		FlagCount;			// Type for counting discontinuities
	typedef typename KernelConfig::LocalFlagCount 	LocalFlagCount;		// Type for local discontinuity counts (that can't overflow)
	typedef typename KernelConfig::SrtsDetails 		SrtsDetails;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running accumulator for the number of discontinuities observed by
	// the CTA over its tile-processing lifetime (managed in each raking thread)
	FlagCount carry;

	// Input and output device pointers
	T *d_in;
	T *d_out;

	// Pool of storage for compacting a tile of values
	T *exchange;

	// Operational details for SRTS scan grid
	SrtsDetails srts_details;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ DownsweepCta(
		SmemStorage &smem_storage,
		T *d_in,
		T *d_out,
		FlagCount spine_partial = 0) :

			srts_details(
				smem_storage.smem_pool.raking_elements,
				smem_storage.warpscan,
				0),
			exchange(smem_storage.smem_pool.exchange),
			d_in(d_in),
			d_out(d_out),
			carry(spine_partial) {}			// Seed carry with spine partial


	/**
	 * Process a single tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FULL_TILE, bool FIRST_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = 0)
	{
		T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];					// Tile of elements
		LocalFlagCount flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];	// Tile of discontinuity flags
		LocalFlagCount ranks[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];	// Tile of local scatter offsets

		// Load data tile, initializing discontinuity flags
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::template Invoke<FIRST_TILE>(
				data, flags, d_in + cta_offset, guarded_elements);

		// Copy discontinuity flags into ranks
		util::io::InitializeTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE>::Copy(ranks, flags);

		// Scan tile of ranks
		LocalFlagCount unique_elements = util::scan::CooperativeTileScan<
			SrtsDetails,
			KernelConfig::LOAD_VEC_SIZE,
			true,							// exclusive
			util::Operators<T>::Sum>::ScanTile(
				srts_details, ranks);

		//
		// Scatter directly without first compacting in smem scratch
		//

		// Scatter valid vertex_id into smem exchange, predicated on flags (treat
		// vertex_id, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER>::Scatter(
				d_out + carry,
				reinterpret_cast<T*>(data),
				reinterpret_cast<LocalFlagCount*>(flags),
				reinterpret_cast<LocalFlagCount*>(ranks));

/*

		//
		// Scatter by first compacting in smem scratch
		//

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Scatter valid data into smem exchange, predicated on flags (treat
		// data, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			util::io::st::NONE>::Scatter(
				exchange,
				reinterpret_cast<T*>(data),
				reinterpret_cast<LocalFlagCount*>(flags),
				reinterpret_cast<LocalFlagCount*>(ranks));

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Gather compacted data from smem exchange (in 1-element stride loads)
		T compacted_data[KernelConfig::TILE_ELEMENTS_PER_THREAD][1];
		util::io::LoadTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelConfig::THREADS,
			util::io::ld::NONE,
			false>::Invoke(								// Guarded loads
					compacted_data, exchange, unique_elements);

		// Scatter compacted data to global output
		util::io::StoreTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			false>::Invoke(								// Guarded stores
				compacted_data, d_out + carry, unique_elements);

		// Barrier sync to protect smem exchange storage
		__syncthreads();
*/

		// Update running discontinuity count for CTA
		carry += unique_elements;
	}
};



} // namespace consecutive_removal
} // namespace b40c

