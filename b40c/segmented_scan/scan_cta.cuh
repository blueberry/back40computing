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

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Derivation of KernelConfig that encapsulates segmented-scan downsweep
 * reduction tile-processing state and routines
 */
template <typename KernelConfig>
struct ScanCta : KernelConfig						// Derive from our config
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef typename KernelConfig::T T;
	typedef typename KernelConfig::Flag Flag;
	typedef typename KernelConfig::SizeT SizeT;
	typedef typename KernelConfig::SrtsSoaDetails SrtsSoaDetails;
	typedef typename KernelConfig::SoaTuple SoaTuple;

	typedef util::Tuple<
		T (*)[KernelConfig::LOAD_VEC_SIZE],
		Flag (*)[KernelConfig::LOAD_VEC_SIZE]> DataSoa;

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

	// Tile of segmented scan elements
	T				partials[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

	// Tile of flags
	Flag			flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ ScanCta(
		SrtsSoaDetails srts_soa_details,
		T 		*d_partials_in,
		Flag 	*d_flags_in,
		T 		*d_partials_out,
		T 		spine_partial = KernelConfig::Identity()) :

			srts_soa_details(srts_soa_details),
			d_partials_in(d_partials_in),
			d_flags_in(d_flags_in),
			d_partials_out(d_partials_out),
			carry(												// Seed carry with spine partial & flag
				spine_partial,
				KernelConfig::FlagIdentity()) {}


	/**
	 * Process a single tile
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		// Load tile of partials
		util::LoadTile<
			T,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			UNGUARDED_IO>::Invoke(partials, d_partials_in, cta_offset, out_of_bounds);

		// Load tile of flags
		util::LoadTile<
			Flag,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			UNGUARDED_IO>::Invoke(flags, d_flags_in, cta_offset, out_of_bounds);

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
		util::StoreTile<
			T,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			UNGUARDED_IO>::Invoke(partials, d_partials_out, cta_offset, out_of_bounds);
	}
};



} // namespace segmented_scan
} // namespace b40c

