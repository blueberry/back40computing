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
 * Spine CTA tuning policy
 */
template <
	int 							_LOG_CTA_THREADS,		// The number of threads per CTA
	int 							_LOG_LOAD_VEC_SIZE,		// The number of consecutive keys to process per thread per global load
	int 							_LOG_LOADS_PER_TILE,	// The number of loads to process per thread per tile
	cub::LoadModifier 				_LOAD_MODIFIER,			// Load cache-modifier
	cub::StoreModifier				_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaScanPassPolicy
{
	enum
	{
		LOG_CTA_THREADS 			= _LOG_CTA_THREADS,
		LOG_LOAD_VEC_SIZE  			= _LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE 			= _LOG_LOADS_PER_TILE,

		CTA_THREADS					= 1 << LOG_CTA_THREADS,
		LOG_TILE_ELEMENTS			= LOG_CTA_THREADS + LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
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
		LOG_CTA_THREADS 				= CtaScanPassPolicy::LOG_CTA_THREADS,
		CTA_THREADS						= 1 << LOG_CTA_THREADS,

		LOG_LOAD_VEC_SIZE  				= CtaScanPassPolicy::LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE					= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 				= CtaScanPassPolicy::LOG_LOADS_PER_TILE,
		LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE,

		LOG_WARP_THREADS 				= cub::DeviceProps::LOG_WARP_THREADS,
		WARP_THREADS					= 1 << LOG_WARP_THREADS,

		LOG_WARPS						= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_CTA_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,
	};

	// CtaPrefixScan utility type
	typedef cub::CtaScan<T, CTA_THREADS, LOADS_PER_TILE> CtaScanT;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaScanT::SmemStorage scan_storage;
		};
	};

private:

	//---------------------------------------------------------------------
	// Thread fields (aggregate state bundle)
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage &cta_smem_storage;

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
	__device__ __forceinline__ CtaScanPass(
		SmemStorage 		&cta_smem_storage,
		T 					*d_in,
		T 					*d_out) :
			cta_smem_storage(cta_smem_storage),
			d_in(d_in),
			d_out(d_out),
			carry(0)
	{}

	/**
	 * Process a single tile
	 */
	template <typename ScanOp, typename SizeT>
	__device__ __forceinline__ void ProcessTile(
		ScanOp scan_op,
		SizeT cta_offset)
	{
		// Tile of scan elements
		T partials[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Load tile
		cub::CtaLoad<CTA_THREADS>::LoadUnguarded(partials, d_in, cta_offset);

		// Scan tile with carry update in raking threads
		T aggregate;
		CtaScanT::ExclusiveScan(
			cta_smem_storage.scan_storage,
			partials,
			partials,
			cub::Sum<T>(),
			0,
			aggregate,
			carry);

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
	template <typename ScanOp, typename SizeT>
	static __device__ __forceinline__ void Scan(
		SmemStorage 		&cta_smem_storage,
		T 					*d_in,
		T 					*d_out,
		ScanOp				scan_op,
		SizeT				&num_elements)
	{
		// Construct state bundle
		CtaScanPass cta(cta_smem_storage, d_in, d_out);

		SizeT cta_offset = 0;
		while (cta_offset + TILE_ELEMENTS <= num_elements)
		{
			// Process full tiles of tile_elements
			ProcessTile(cta_offset, scan_op);

			cta_offset += TILE_ELEMENTS;
		}
	}

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
