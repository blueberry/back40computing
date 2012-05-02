/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * CTA-processing functionality for scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace radix_sort {
namespace spine {


template <
	typename KernelPolicy,
	typename T,
	typename SizeT>
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions
	//---------------------------------------------------------------------

	struct SmemStorage
	{
	};

	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage &smem_storage;

	// Running partial accumulated by the CTA over its tile-processing
	// lifetime (managed in each raking thread)
	T carry;

	// Input and output device pointers
	T *d_in;
	T *d_out;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 		&smem_storage,
		T 					*d_in,
		T 					*d_out) :
			// Initializers
			smem_storage(smem_storage),
			d_in(d_in),
			d_out(d_out),
			carry(0)
	{}

	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(SizeT cta_offset)
	{
		// Tile of scan elements
		T partials[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				partials,
				d_in,
				cta_offset);


		// Store tile
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			false>::Store(
				partials,
				d_out,
				cta_offset);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		SizeT num_elements)
	{
		for (SizeT cta_offset = 0;
			cta_offset < num_elements;
			cta_offset += KernelPolicy::TILE_ELEMENTS)
		{
			// Process full tiles of tile_elements
			ProcessTile(cta_offset);
		}
	}

};


} // namespace spine
} // namespace radix_sort
} // namespace b40c

