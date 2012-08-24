/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * "Spine-scan" CTA abstraction for scanning radix digit histograms
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------


/**
 * Spine scan CTA tuning policy
 */
template <
	int 							_CTA_THREADS,			// The number of threads per CTA
	int 							_THREAD_STRIP_ITEMS,	// The number of consecutive keys to process per thread per global load
	int 							_TILE_STRIPS,			// The number of loads to process per thread per tile
	cub::LoadModifier 				_LOAD_MODIFIER,			// Load cache-modifier
	cub::StoreModifier				_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaScanPassPolicy
{
	enum
	{
		CTA_THREADS 				= _CTA_THREADS,
		THREAD_STRIP_ITEMS  		= _THREAD_STRIP_ITEMS,
		TILE_STRIPS 				= _TILE_STRIPS,
		TILE_ITEMS				= CTA_THREADS * THREAD_STRIP_ITEMS * TILE_STRIPS,
	};

	static const cub::LoadModifier 		LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const cub::StoreModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig	SMEM_CONFIG			= _SMEM_CONFIG;
};


//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------


/**
 * CTA-wide abstraction for computing a prefix scan over a range of input tiles
 */
template <
	typename CtaScanPassPolicy,
	typename T>
class CtaScanPass
{
private:

	//---------------------------------------------------------------------
	// Constants and type definitions
	//---------------------------------------------------------------------

	enum
	{
		CTA_THREADS					= CtaScanPassPolicy::CTA_THREADS,
		THREAD_STRIP_ITEMS			= CtaScanPassPolicy::THREAD_STRIP_ITEMS,
		TILE_STRIPS					= CtaScanPassPolicy::TILE_STRIPS,
		TILE_ITEMS				= CtaScanPassPolicy::TILE_ITEMS,
	};

	// CtaScan utility type
	typedef cub::CtaScan<T, CTA_THREADS, TILE_STRIPS> CtaScanT;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage : CtaScanT::SmemStorage {};

private:

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Process a single tile
	 */
	template <typename SizeT>
	static __device__ __forceinline__ void ProcessTile(
		SmemStorage 	&smem_storage,
		T 				*d_in,
		T 				*d_out,
		SizeT 			cta_offset,
		T				&carry)
	{
		// Tile of scan elements
		T partials[TILE_STRIPS][THREAD_STRIP_ITEMS];

		// Load tile
		cub::CtaLoad<CTA_THREADS>::LoadUnguarded(partials, d_in, cta_offset);

		// Scan tile with carry in thread-0
		T aggregate;
		CtaScanT::ExclusiveSum(smem_storage, partials, partials, aggregate, carry);

		// Store tile
		cub::CtaStore<CTA_THREADS>::StoreUnguarded(partials, d_out, cta_offset);
	}

public:

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scan a range of input tiles
	 */
	template <typename SizeT>
	static __device__ __forceinline__ void ScanPass(
		SmemStorage 		&smem_storage,
		T 					*d_in,
		T 					*d_out,
		SizeT				&num_elements)
	{
		// Running partial accumulated by the CTA over its tile-processing
		// lifetime (managed in each raking thread)
		T carry = 0;

		SizeT cta_offset = 0;
		while (cta_offset + TILE_ITEMS <= num_elements)
		{
			// Process full tiles of tile_elements
			ProcessTile(smem_storage, d_in, d_out, cta_offset, carry);

			cta_offset += TILE_ITEMS;
		}
	}

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
