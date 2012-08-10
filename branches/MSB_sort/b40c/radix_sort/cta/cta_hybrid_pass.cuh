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

#include "../../util/basic_utils.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_umbrella.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta_block.cuh"
#include "../../radix_sort/cta_radix_sort.cuh"
#include "../../radix_sort/cta_upsweep.cuh"
#include "../../radix_sort/cta_downsweep.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/**
 * Hybrid CTA tuning policy.
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_CTA_THREADS,			// The number of threads per CTA
	int 							_THREAD_ELEMENTS,		// The number of consecutive keys to process per thread per tile
	util::io::ld::CacheModifier	 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaHybridPolicy
{
	enum
	{
		RADIX_BITS					= _RADIX_BITS,
		MIN_CTA_OCCUPANCY  			= _MIN_CTA_OCCUPANCY,
		CTA_THREADS 				= _CTA_THREADS,
		THREAD_ELEMENTS 			= _THREAD_ELEMENTS,

		TILE_ELEMENTS				= CTA_THREADS * THREAD_ELEMENTS,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
};



/**
 * "Hybrid" CTA abstraction for locally sorting small blocks or performing
 * global distribution passes over large blocks
 */
template <
	typename CtaHybridPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
struct CtaHybrid
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	enum
	{
		PASS_RADIX_BITS		= CtaHybridPolicy::Upsweep::RADIX_BITS,
		PASS_RADIX_DIGITS 	= 1 << PASS_RADIX_BITS,
	};

	// Single-tile CTA abstraction
	typedef CtaSingle<
		typename CtaHybridPolicy::Upsweep,
		KeyType,
		ValueType> SingleCta;

	// Upsweep CTA abstraction
	typedef CtaUpsweep<
		typename CtaHybridPolicy::Upsweep,
		SizeT,
		KeyType> UpsweepCta;

	// Downsweep CTA abstraction
	typedef CtaDownsweep<
		typename CtaHybridPolicy::Downsweep,
		SizeT,
		KeyType,
		ValueType> DownsweepCta;


	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		BinDescriptor								partition;

		union
		{
			typename BlockCta::SmemStorage 		block_storage;
			typename UpsweepCta::SmemStorage 	upsweep_storage;
			typename DownsweepCta::SmemStorage 	downsweep_storage;
			volatile SizeT						warpscan[2][PASS_RADIX_DIGITS];
		};
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Process work range
	 */
	static __device__ __forceinline__ void ProcessWorkRange(
		BinDescriptor		*d_bins_in,
		BinDescriptor		*d_bins_out,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		KeyType 		*d_keys_final,
		ValueType 		*d_values_in,
		ValueType 		*d_values_out,
		ValueType 		*d_values_final,
		int				low_bit)
	{
		// Retrieve work
		if (threadIdx.x == 0)
		{
			cta_smem_storage.partition = d_bins_in[blockIdx.x];
/*
			printf("\tCTA %d loaded partition (low bit %d, current bit %d) of %d elements at offset %d\n",
				blockIdx.x,
				low_bit,
				cta_smem_storage.partition.current_bit,
				cta_smem_storage.partition.num_elements,
				cta_smem_storage.partition.offset);
*/

			// Reset current partition descriptor
			d_bins_in[blockIdx.x].num_elements = 0;
		}

		__syncthreads();

		// Quit if there is no work
		if (cta_smem_storage.partition.num_elements == 0) return;

		// Choose whether to block-sort or pass-sort
		if (cta_smem_storage.partition.num_elements < TILE_ELEMENTS)
		{
			// Perform block sort
			BlockCta::Sort(
				cta_smem_storage.block_storage,
				d_keys_in,
				d_keys_final,
				d_values_in,
				d_values_final,
				low_bit,
				cta_smem_storage.partition.current_bit - low_bit,
				cta_smem_storage.partition.offset,
				cta_smem_storage.partition.num_elements);

			// Output new (dummy) partition descriptors
			if (threadIdx.x < PASS_RADIX_DIGITS)
			{
				BinDescriptor partition(0, 0, 0);
				SizeT partition_offset = (blockIdx.x * PASS_RADIX_DIGITS) + threadIdx.x;
				d_bins_out[partition_offset] = partition;
			}
		}
		else
		{
			// Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
			SizeT bin_count;
			UpsweepCta::Upsweep(
				cta_smem_storage.upsweep_storage,
				d_keys_in,
				cta_smem_storage.partition.current_bit,
				cta_smem_storage.partition.offset,
				cta_smem_storage.partition.out_of_bounds,
				bin_count);

			__syncthreads();

			// Exclusive scan across bin counts
			SizeT bin_prefix;
			if (threadIdx.x < PASS_RADIX_DIGITS)
			{
				// Initialize warpscan identity regions
				warpscan[0][threadIdx.x] = 0;

				// Warpscan
				SizeT partial = bin_count;
				warpscan[1][threadIdx.x] = partial;

				#pragma unroll
				for (int STEP = 0; STEP < LOG_WARP_THREADS; STEP++)
				{
					partial += warpscan[1][threadIdx.x - (1 << STEP)];
					warpscan[1][threadIdx.x] = partial;
				}

				bin_prefix = partial - bin_count;
			}

			// Output new partition descriptors
			if (threadIdx.x < PASS_RADIX_DIGITS)
			{
				BinDescriptor partition(
					bin_prefix,
					bin_count,
					cta_smem_storage.partition.current_bit - PASS_RADIX_DIGITS);

				SizeT partition_offset = (blockIdx.x * PASS_RADIX_DIGITS) + threadIdx.x;
				d_bins_out[partition_offset] = partition;
			}


			// Note: no syncthreads() necessary

			// Distribute keys
			DownsweepCta::Downsweep(
				cta_smem_storage.downsweep_storage,
				d_in_keys,
				d_out_keys,
				d_in_values,
				d_out_values,
				current_bit,
				bin_prefix);
		}
	}

};






} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
