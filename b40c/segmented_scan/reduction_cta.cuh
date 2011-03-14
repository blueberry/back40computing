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

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/reduction/soa/cooperative_soa_reduction.cuh>

namespace b40c {
namespace segmented_scan {


/**
 * Derivation of KernelConfig that encapsulates segmented-scan upsweep
 * reduction tile-processing state and routines
 */
template <typename KernelConfig>
struct ReductionCta : KernelConfig						// Derive from our config
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename ReductionCta::T T;
	typedef typename ReductionCta::Flag Flag;
	typedef typename ReductionCta::SizeT SizeT;
	typedef typename ReductionCta::SrtsSoaDetails SrtsSoaDetails;
	typedef typename ReductionCta::SoaTuple SoaTuple;

	typedef T PartialsTile[ReductionCta::LOADS_PER_TILE][ReductionCta::LOAD_VEC_SIZE];
	typedef Flag FlagsTile[ReductionCta::LOADS_PER_TILE][ReductionCta::LOAD_VEC_SIZE];

	typedef util::Tuple<PartialsTile&, FlagsTile&> DataSoa;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Operational details for SRTS grid
	SrtsSoaDetails srts_soa_details;

	// The tuple value we will accumulate (in SrtsDetails::CUMULATIVE_THREAD thread only)
	SoaTuple carry;

	// Input device pointers
	T 		*d_partials_in;
	Flag 	*d_flags_in;

	// Spine output device pointers
	T 		*d_spine_partial;
	Flag 	*d_spine_flag;

	// Tile of segmented scan elements
	PartialsTile	partials;

	// Tile of flags
	FlagsTile		flags;

	// Soa reference to tiles
	DataSoa			data_soa;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ ReductionCta(
		const SrtsSoaDetails &srts_soa_details,
		T 		*d_partials_in,
		Flag 	*d_flags_in,
		T 		*d_spine_partial,
		Flag 	*d_spine_flag) :

			srts_soa_details(srts_soa_details),
			d_partials_in(d_partials_in),
			d_flags_in(d_flags_in),
			d_spine_partial(d_spine_partial),
			d_spine_flag(d_spine_flag),
			data_soa(partials, flags),
			carry(ReductionCta::SoaIdentity()) {}


	/**
	 * Load transform function for assigning identity to partial values
	 * that are out of range.
	 */
	static __device__ __forceinline__ void LoadTransformT(
		T &val,
		bool in_bounds)
	{
		// Assigns identity value to out-of-bounds loads
		if (!in_bounds) val = ReductionCta::Identity();
	}


	/**
	 * Load transform function for assigning identity to flag values
	 * that are out of range.
	 */
	static __device__ __forceinline__ void LoadTransformFlag(
		Flag &val,
		bool in_bounds)
	{
		// Assigns identity value to out-of-bounds loads
		if (!in_bounds) val = ReductionCta::FlagIdentity();
	}


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
			ReductionCta::LOG_LOADS_PER_TILE,
			ReductionCta::LOG_LOAD_VEC_SIZE,
			ReductionCta::THREADS,
			ReductionCta::READ_MODIFIER,
			UNGUARDED_IO,
			LoadTransformT>::Invoke(partials, d_partials_in, cta_offset, out_of_bounds);

		// Load tile of flags
		util::LoadTile<
			T,
			SizeT,
			ReductionCta::LOG_LOADS_PER_TILE,
			ReductionCta::LOG_LOAD_VEC_SIZE,
			ReductionCta::THREADS,
			ReductionCta::READ_MODIFIER,
			UNGUARDED_IO,
			LoadTransformFlag>::Invoke(flags, d_flags_in, cta_offset, out_of_bounds);

		// Reduce tile with carry
		util::reduction::soa::CooperativeSoaTileReduction<
			SrtsSoaDetails,
			ReductionCta::LOAD_VEC_SIZE,
			ReductionCta::SoaScanOp>::template ReduceTileWithCarry<DataSoa>(
				srts_soa_details, data_soa);

		// Barrier to protect srts_soa_details before next tile
		__syncthreads();
	}


	/**
	 * Stores final reduction to output
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		// Write output
		if (threadIdx.x == SrtsDetails::CUMULATIVE_THREAD) {
			util::ModifiedStore<T, ReductionCta::WRITE_MODIFIER>::St(carry.t0, d_spine_partial, 0);
			util::ModifiedStore<T, ReductionCta::WRITE_MODIFIER>::St(carry.t1, d_spine_flag, 0);
		}
	}

};



} // namespace segmented_scan
} // namespace b40c

