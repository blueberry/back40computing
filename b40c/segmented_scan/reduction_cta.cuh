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
	T 				*d_spine_partial;
	Flag 			*d_spine_flag;

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
	__device__ __forceinline__ ReductionCta(
		SrtsSoaDetails 	srts_soa_details,
		T 				*d_partials_in,
		Flag 			*d_flags_in,
		T 				*d_spine_partial,
		Flag 			*d_spine_flag) :

			srts_soa_details(srts_soa_details),
			d_partials_in(d_partials_in),
			d_flags_in(d_flags_in),
			d_spine_partial(d_spine_partial),
			d_spine_flag(d_spine_flag),
			carry(KernelConfig::SoaTupleIdentity()) {}


	/**
	 * Load transform function for assigning identity to partial values
	 * that are out of range.
	 */
	static __device__ __forceinline__ void LoadTransformT(
		T &val,
		bool in_bounds)
	{
		// Assigns identity value to out-of-bounds loads
		if (!in_bounds) val = KernelConfig::Identity();
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
		if (!in_bounds) val = KernelConfig::FlagIdentity();
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
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			UNGUARDED_IO,
			LoadTransformT>::Invoke(partials, d_partials_in, cta_offset, out_of_bounds);

		// Load tile of flags
		util::LoadTile<
			Flag,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			UNGUARDED_IO,
			LoadTransformFlag>::Invoke(flags, d_flags_in, cta_offset, out_of_bounds);

		// Reduce tile with carry
		util::reduction::soa::CooperativeSoaTileReduction<
			SrtsSoaDetails,
			KernelConfig::LOAD_VEC_SIZE,
			KernelConfig::SoaScanOp>::template ReduceTileWithCarry<DataSoa>(
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
			util::ModifiedStore<T, KernelConfig::WRITE_MODIFIER>::St(carry.t0, d_spine_partial, 0);
			util::ModifiedStore<Flag, KernelConfig::WRITE_MODIFIER>::St(carry.t1, d_spine_flag, 0);
		}
	}
};



} // namespace segmented_scan
} // namespace b40c

