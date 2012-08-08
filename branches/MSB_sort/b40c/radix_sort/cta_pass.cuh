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
 *
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_umbrella.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta_radix_sort.cuh"
#include "../../radix_sort/cta_tile.cuh"
#include "../../radix_sort/cta_upsweep.cuh"
#include "../../radix_sort/cta_downsweep.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace partition {


/**
 * CtaPass tuning policy.
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_CTA_THREADS,			// The number of threads per CTA
	int 							_THREAD_ELEMENTS,		// The number of consecutive keys to process per thread per tile
	util::io::ld::CacheModifier	 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaPassPolicy
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
 *
 */
template <
	typename CtaPassPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
struct CtaPass
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	enum
	{
		PASS_RADIX_BITS		= CtaPassPolicy::Upsweep::RADIX_BITS,
		PASS_RADIX_DIGITS 	= 1 << PASS_RADIX_BITS,
	};

	// Block CTA abstraction
	typedef CtaPass<
		typename CtaPassPolicy::Upsweep,
		KeyType,
		ValueType> BlockCta;

	// Upsweep CTA abstraction
	typedef CtaUpsweep<
		typename CtaPassPolicy::Upsweep,
		SizeT,
		KeyType> UpsweepCta;

	// Downsweep CTA abstraction
	typedef CtaDownsweep<
		typename CtaPassPolicy::Downsweep,
		SizeT,
		KeyType,
		ValueType> DownsweepCta;


	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		Partition								partition;

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
		Partition		*d_partitions_in,
		Partition		*d_partitions_out,
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
			smem_storage.partition = d_partitions_in[blockIdx.x];
/*
			printf("\tCTA %d loaded partition (low bit %d, current bit %d) of %d elements at offset %d\n",
				blockIdx.x,
				low_bit,
				smem_storage.partition.current_bit,
				smem_storage.partition.num_elements,
				smem_storage.partition.offset);
*/

			// Reset current partition descriptor
			d_partitions_in[blockIdx.x].num_elements = 0;
		}

		__syncthreads();

		// Quit if there is no work
		if (smem_storage.partition.num_elements == 0) return;

		// Choose whether to block-sort or pass-sort
		if (smem_storage.partition.num_elements < TILE_ELEMENTS)
		{
			// Perform block sort
			BlockCta::Sort(
				smem_storage.block_storage,
				d_keys_in,
				d_keys_final,
				d_values_in,
				d_values_final,
				low_bit,
				smem_storage.partition.current_bit - low_bit,
				smem_storage.partition.offset,
				smem_storage.partition.num_elements);

			// Output new (dummy) partition descriptors
			if (threadIdx.x < PASS_RADIX_DIGITS)
			{
				Partition partition(0, 0, 0);
				SizeT partition_offset = (blockIdx.x * PASS_RADIX_DIGITS) + threadIdx.x;
				d_partitions_out[partition_offset] = partition;
			}
		}
		else
		{
			// Compute bin-count for each radix digit (valid in tid < RADIX_DIGITS)
			SizeT bin_count;
			UpsweepCta::Upsweep(
				smem_storage.upsweep_storage,
				d_keys_in,
				smem_storage.partition.current_bit,
				smem_storage.partition.offset,
				smem_storage.partition.out_of_bounds,
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
				Partition partition(
					bin_prefix,
					bin_count,
					smem_storage.partition.current_bit - PASS_RADIX_DIGITS);

				SizeT partition_offset = (blockIdx.x * PASS_RADIX_DIGITS) + threadIdx.x;
				d_partitions_out[partition_offset] = partition;
			}


			// Note: no syncthreads() necessary

			// Distribute keys
			DownsweepCta::Downsweep(
				smem_storage.downsweep_storage,
				d_in_keys,
				d_out_keys,
				d_in_values,
				d_out_values,
				current_bit,
				bin_prefix);
		}
	}

};



/**
 * Kernel entry point
 */
template <
	typename CtaPassPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (CtaPassPolicy::CTA_THREADS, CtaPassPolicy::MIN_CTA_OCCUPANCY)
__global__
void Kernel(
	Partition							*d_partitions_in,
	Partition							*d_partitions_out,
	KeyType 							*d_keys_in,
	KeyType 							*d_keys_out,
	KeyType 							*d_keys_final,
	ValueType 							*d_values_in,
	ValueType 							*d_values_out,
	ValueType 							*d_values_final,
	int									low_bit)
{
	// CTA abstraction type
	typedef CtaPass<CtaPassPolicy, SizeT, KeyType, ValueType> CtaPass;

	// Shared memory pool
	__shared__ typename CtaPass::SmemStorage smem_storage;

	CtaPass::ProcessWorkRange(
		smem_storage,
		d_partitions_in,
		d_partitions_out,
		d_keys_in,
		d_keys_out,
		d_keys_final,
		d_values_in,
		d_values_out,
		d_values_final,
		low_bit);
}




} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
