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
 * "Hybrid" CTA abstraction for locally sorting small blocks or performing
 * global distribution passes over large blocks
 ******************************************************************************/

#pragma once

#include "../../cub/cub.cuh"
#include "cta_single_tile.cuh"
#include "cta_upsweep_pass.cuh"
#include "cta_downsweep_pass.cuh"
#include "../ns_wrapper.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {

//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------


/**
 * Hybrid CTA tuning policy
 */
template <
	int 						RADIX_BITS,						// The number of radix bits, i.e., log2(bins)
	int 						CTA_THREADS,					// The number of threads per CTA

	int 						UPSWEEP_THREAD_ITEMS,			// The number of consecutive upsweep items to load per thread per tile
	int 						DOWNSWEEP_THREAD_ITEMS,			// The number of consecutive downsweep items to load per thread per tile
	ScatterStrategy 			DOWNSWEEP_SCATTER_STRATEGY,		// Downsweep strategy
	int 						SINGLE_TILE_THREAD_ITEMS,		// The number of consecutive single-tile items to load per thread per tile

	cub::LoadModifier 			LOAD_MODIFIER,					// Load cache-modifier
	cub::StoreModifier			STORE_MODIFIER,					// Store cache-modifier
	cudaSharedMemConfig			SMEM_CONFIG>					// Shared memory bank size
struct CtaHybridPassPolicy
{
	// Upsweep pass policy
	typedef CtaUpsweepPassPolicy<
		RADIX_BITS,
		CTA_THREADS,
		UPSWEEP_THREAD_ITEMS,
		LOAD_MODIFIER,
		STORE_MODIFIER,
		SMEM_CONFIG> CtaUpsweepPassPolicyT;

	// Downsweep pass policy
	typedef CtaDownsweepPassPolicy<
		RADIX_BITS,
		CTA_THREADS,
		DOWNSWEEP_THREAD_ITEMS,
		DOWNSWEEP_SCATTER_STRATEGY,
		LOAD_MODIFIER,
		STORE_MODIFIER,
		SMEM_CONFIG> CtaDownsweepPassPolicyT;

	// Single tile policy
	typedef CtaSingleTilePolicy<
		RADIX_BITS,
		CTA_THREADS,
		SINGLE_TILE_THREAD_ITEMS,
		LOAD_MODIFIER,
		STORE_MODIFIER,
		SMEM_CONFIG> CtaSingleTilePolicyT;
};



//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------


/**
 * "Hybrid" CTA abstraction for locally sorting small blocks or performing
 * global distribution passes over large blocks
 */
template <
	typename CtaHybridPassPolicy,
	typename KeyType,
	typename ValueType,
	typename SizeT>
class CtaHybridPass
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	enum
	{
		CTA_THREADS			= CtaHybridPassPolicy::CtaUpsweepPassPolicyT::CTA_THREADS,

		RADIX_DIGITS		= 1 << CtaHybridPassPolicy::CtaUpsweepPassPolicyT::RADIX_BITS,

		WARP_THREADS		= cub::DeviceProps::WARP_THREADS,

		SINGLE_TILE_ITEMS	= CtaHybridPassPolicy::CtaSingleTilePolicyT::TILE_ITEMS,
	};

	// CTA upsweep abstraction
	typedef CtaUpsweepPass<
		typename CtaHybridPassPolicy::CtaUpsweepPassPolicyT,
		KeyType,
		SizeT> CtaUpsweepPassT;

	// CTA downsweep abstraction
	typedef CtaDownsweepPass<
		typename CtaHybridPassPolicy::CtaDownsweepPassPolicyT,
		KeyType,
		ValueType,
		SizeT> CtaDownsweepPassT;

	// CTA single-tile abstraction
	typedef CtaSingleTile<
		typename CtaHybridPassPolicy::CtaSingleTilePolicyT,
		KeyType,
		ValueType> CtaSingleTileT;

	// Warp scan abstraction
	typedef cub::WarpScan<
		SizeT,
		1,
		RADIX_DIGITS> WarpScanT;

	// CTA scan abstraction
	typedef cub::CtaScan<
		SizeT,
		CTA_THREADS> CtaScanT;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaUpsweepPassT::SmemStorage 		upsweep_storage;
			typename CtaDownsweepPassT::SmemStorage 	downsweep_storage;
			typename CtaSingleTileT::SmemStorage 		single_storage;
			typename WarpScanT::SmemStorage				warp_scan_storage;
			typename CtaScanT::SmemStorage				cta_scan_storage;
		};
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Process work range
	 */
	static __device__ __forceinline__ void Sort(
		SmemStorage 	&cta_smem_storage,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		KeyType 		*d_keys_final,
		ValueType 		*d_values_in,
		ValueType 		*d_values_out,
		ValueType 		*d_values_final,
		int				current_bit,
		int				low_bit,
		SizeT			num_elements,
		int				&bin_count,			// The digit count for tid'th bin (output param, valid in the first RADIX_DIGITS threads)
		int				&bin_prefix)		// The base offset for each digit (valid in the first RADIX_DIGITS threads)
	{
		bin_count = 0;
		bin_prefix = 0;

		// Choose whether to block-sort or pass-sort
		if (num_elements < SINGLE_TILE_ITEMS)
		{
			// Perform block sort
			CtaSingleTileT::Sort(
				cta_smem_storage.single_storage,
				d_keys_in,
				d_keys_final,
				d_values_in,
				d_values_final,
				low_bit,
				cta_smem_storage.partition.current_bit - low_bit,
				cta_smem_storage.partition.offset,
				cta_smem_storage.partition.num_elements);
		}
		else
		{
			// Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
			SizeT bin_count;
			CtaUpsweepPassT::UpsweepPass(
				cta_smem_storage.upsweep_storage,
				d_keys_in,
				current_bit,
				num_elements,
				bin_count);

			__syncthreads();

			// Prefix sum over bin counts
			SizeT bin_prefix;
			if (RADIX_DIGITS <= WARP_THREADS)
			{
				// Warp prefix sum
				if (threadIdx.x < RADIX_DIGITS)
				{
					WarpScanT::ExclusiveSum(cta_smem_storage.warp_scan_storage, bin_count, bin_prefix);
				}
			}
			else
			{
				// Cta prefix sum
				CtaScanT::ExclusiveSum(cta_smem_storage.cta_scan_storage, bin_count, bin_prefix);
			}

			// Note: no syncthreads() necessary

			// Distribute keys
			CtaDownsweepPassT::DownsweepPass(
				cta_smem_storage.downsweep_storage,
				bin_prefix,
				d_keys_in,
				d_keys_out,
				d_values_in,
				d_values_out,
				current_bit,
				num_elements);
		}
	}
};






} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
