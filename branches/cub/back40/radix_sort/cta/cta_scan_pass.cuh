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

#include "../../util/cuda_properties.cuh"
#include "../../util/srts_grid.cuh"
#include "../../util/srts_details.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/io/load_tile.cuh"
#include "../../util/io/store_tile.cuh"
#include "../../util/scan/cooperative_scan.cuh"
#include "../../util/ns_wrapper.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace cta {


/**
 * Spine CTA tuning policy
 */
template <
	int 							_LOG_CTA_THREADS,		// The number of threads per CTA
	int 							_LOG_LOAD_VEC_SIZE,		// The number of consecutive keys to process per thread per global load
	int 							_LOG_LOADS_PER_TILE,	// The number of loads to process per thread per tile
	util::io::ld::CacheModifier 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
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

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
};



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

		LOG_WARP_THREADS 				= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS					= 1 << LOG_WARP_THREADS,

		LOG_WARPS						= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS							= 1 << LOG_WARPS,

		LOG_TILE_ELEMENTS_PER_THREAD	= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD,

		LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + LOG_CTA_THREADS,
		TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS,
	};

	/**
	 * Raking grid type
	 */
	typedef util::RakingGrid<
		T,										// Partial type
		LOG_CTA_THREADS,						// Depositing threads (the CTA size)
		LOG_LOADS_PER_TILE,						// Lanes (the number of loads)
		LOG_WARP_THREADS,						// 1 warp of raking threads
		true>									// There are prefix dependences between lanes
			RakingGrid;

	/**
	 * Operational details type for raking grid type
	 */
	typedef util::RakingDetails<RakingGrid> RakingDetails;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		T warpscan[2][WARP_THREADS];
		T raking_elements[RakingGrid::TOTAL_RAKING_ELEMENTS];		// Raking raking elements
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

	// Operational details for raking scan grid
	RakingDetails raking_details;

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
			// Initializers
			raking_details(
				cta_smem_storage.raking_elements,
				cta_smem_storage.warpscan,
				0),
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
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			CTA_THREADS,
			CtaScanPassPolicy::LOAD_MODIFIER,
			false>::LoadValid(
				partials,
				d_in,
				cta_offset);

		// Scan tile with carry update in raking threads
		util::scan::CooperativeTileScan<
			LOAD_VEC_SIZE,
			true>::ScanTileWithCarry(
				raking_details,
				partials,
				carry,
				scan_op);

		// Store tile
		util::io::StoreTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			CTA_THREADS,
			CtaScanPassPolicy::STORE_MODIFIER,
			false>::Store(
				partials,
				d_out,
				cta_offset);
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
		ScanCta cta(cta_smem_storage, d_keys_in, current_bit);

		SizeT cta_offset = 0;
		while (cta_offset + TILE_ELEMENTS <= num_elements)
		{
			// Process full tiles of tile_elements
			ProcessTile(cta_offset, scan_op);

			cta_offset += TILE_ELEMENTS;
		}
	}

};


} // namespace cta
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
