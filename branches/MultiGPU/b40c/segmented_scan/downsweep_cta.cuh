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
 * Tile-processing functionality for segmented scan downsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Derivation of KernelConfig that encapsulates segmented-scan downsweep
 * reduction tile-processing state and routines
 */
template <typename KernelConfig>
struct DownsweepCta : KernelConfig						// Derive from our config
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef typename KernelConfig::T 				T;
	typedef typename KernelConfig::Flag 			Flag;
	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::SrtsSoaDetails 	SrtsSoaDetails;
	typedef typename KernelConfig::SoaTuple 		SoaTuple;

	typedef util::Tuple<
		T (*)[KernelConfig::LOAD_VEC_SIZE],
		Flag (*)[KernelConfig::LOAD_VEC_SIZE]> 		DataSoa;

	// This kernel can only operate in inclusive scan mode if the it's the final kernel
	// in the scan pass
	static const bool KERNEL_EXCLUSIVE = (!KernelConfig::FINAL_KERNEL || KernelConfig::EXCLUSIVE);

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
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ DownsweepCta(
		SmemStorage 	&smem_storage,
		T 				*d_partials_in,
		Flag 			*d_flags_in,
		T 				*d_partials_out,
		T 				spine_partial = KernelConfig::Identity()) :

			srts_soa_details(
				typename SrtsSoaDetails::GridStorageSoa(
					smem_storage.smem_pool.raking_elements.partials_raking_elements,
					smem_storage.smem_pool.raking_elements.flags_raking_elements),
				typename SrtsSoaDetails::WarpscanSoa(
					smem_storage.partials_warpscan,
					smem_storage.flags_warpscan),
				KernelConfig::SoaTupleIdentity()),
			d_partials_in(d_partials_in),
			d_flags_in(d_flags_in),
			d_partials_out(d_partials_out),
			carry(												// Seed carry with spine partial & flag
				spine_partial,
				KernelConfig::FlagIdentity()) {}


	/**
	 * Process a single tile
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = 0)
	{
		// Tiles of segmented scan elements and flags
		T				partials[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];
		Flag			flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

		// Load tile of partials
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::Invoke(partials, d_partials_in + cta_offset, guarded_elements);

		// Load tile of flags
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::Invoke(flags, d_flags_in + cta_offset, guarded_elements);

		// Scan tile with carry update in raking threads
		util::scan::soa::CooperativeSoaTileScan<
			SrtsSoaDetails,
			KernelConfig::LOAD_VEC_SIZE,
			KERNEL_EXCLUSIVE,
			KernelConfig::SoaScanOp,
			KernelConfig::FinalSoaScanOp>::template ScanTileWithCarry<DataSoa>(
				srts_soa_details,
				DataSoa(partials, flags),
				carry);

		// Store tile of partials
		util::io::StoreTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			FULL_TILE>::Invoke(partials, d_partials_out + cta_offset, guarded_elements);
	}
};



} // namespace segmented_scan
} // namespace b40c

