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
 * Tile-processing functionality for segmented scan upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/reduction/soa/cooperative_soa_reduction.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Derivation of KernelConfig that encapsulates segmented-scan upsweep
 * reduction tile-processing state and routines
 */
template <typename KernelConfig>
struct UpsweepCta : KernelConfig						// Derive from our config
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::T T;
	typedef typename KernelConfig::Flag Flag;
	typedef typename KernelConfig::SizeT SizeT;
	typedef typename KernelConfig::SrtsSoaDetails SrtsSoaDetails;
	typedef typename KernelConfig::SoaTuple SoaTuple;

	typedef util::Tuple<
		T (*)[KernelConfig::LOAD_VEC_SIZE],
		Flag (*)[KernelConfig::LOAD_VEC_SIZE]> DataSoa;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails 	srts_soa_details;

	// The tuple value we will accumulate (in SrtsDetails::CUMULATIVE_THREAD thread only)
	SoaTuple 		carry;

	// Input device pointers
	T 				*d_partials_in;
	Flag 			*d_flags_in;

	// Spine output device pointers
	T 				*d_spine_partials;
	Flag 			*d_spine_flags;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ UpsweepCta(
		SmemStorage 	&smem_storage,
		T 				*d_partials_in,
		Flag 			*d_flags_in,
		T 				*d_spine_partials,
		Flag 			*d_spine_flags) :

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
			d_spine_partials(d_spine_partials),
			d_spine_flags(d_spine_flags),
			carry(KernelConfig::SoaTupleIdentity()) {}

	/**
	 * Process a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
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
			true>::Invoke(						// unguarded I/O
				partials, d_partials_in + cta_offset);

		// Load tile of flags
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::Invoke(						// unguarded I/O
				flags, d_flags_in + cta_offset);

		// Reduce tile with carry maintained by thread SrtsSoaDetails::CUMULATIVE_THREAD
		util::reduction::soa::CooperativeSoaTileReduction<
			SrtsSoaDetails,
			KernelConfig::LOAD_VEC_SIZE,
			KernelConfig::SoaScanOp>::template ReduceTileWithCarry<true, DataSoa>(
				srts_soa_details,
				DataSoa(partials, flags),
				carry);

		// Barrier to protect srts_soa_details before next tile
		__syncthreads();
	}


	/**
	 * Stores final reduction to output
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		// Write output
		if (threadIdx.x == SrtsSoaDetails::CUMULATIVE_THREAD) {

			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
				carry.t0, d_spine_partials + blockIdx.x);

			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
				carry.t1, d_spine_flags + blockIdx.x);
		}
	}
};



} // namespace segmented_scan
} // namespace b40c

