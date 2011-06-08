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
 * Abstract downsweep CTA processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

namespace b40c {
namespace partition {
namespace downsweep {


/**
 * Cta
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedCta,									// Derived CTA class
	template <typename Policy> class Tile>			// Derived Tile class to use
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;
	typedef typename KernelPolicy::SmemStorage				SmemStorage;
	typedef typename KernelPolicy::Grid::LanePartial		LanePartial;

	typedef DerivedCta Dispatch;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	typename KernelPolicy::SmemStorage 	&smem_storage;

	// Input and output device pointers
	KeyType								*&d_in_keys;
	KeyType								*&d_out_keys;

	ValueType							*&d_in_values;
	ValueType							*&d_out_values;

	SizeT								*&d_spine;

	// SRTS details
	LanePartial							base_composite_counter;
	int									*raking_segment;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*&d_in_keys,
		KeyType 		*&d_out_keys,
		ValueType 		*&d_in_values,
		ValueType 		*&d_out_values,
		SizeT 			*&d_spine,
		LanePartial		base_composite_counter,
		int				*raking_segment) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_out_keys(d_out_keys),
			d_in_values(d_in_values),
			d_out_values(d_out_values),
			d_spine(d_spine),
			base_composite_counter(base_composite_counter),
			raking_segment(raking_segment)
	{
		if (threadIdx.x < KernelPolicy::BINS) {

			// Reset value-area of bin_warpscan
			smem_storage.bin_warpscan[1][threadIdx.x] = 0;

			// Read bin_carry in parallel
			SizeT my_bin_carry;
			int spine_bin_offset = util::FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
			util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(
				my_bin_carry,
				d_spine + spine_bin_offset);

			smem_storage.bin_carry[threadIdx.x] = my_bin_carry;
		}
	}


	/**
	 * Process tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<KernelPolicy> tile;

		tile.Partition(
			cta_offset,
			guarded_elements,
			(Dispatch *) this);
	}
};


} // namespace downsweep
} // namespace partition
} // namespace b40c

