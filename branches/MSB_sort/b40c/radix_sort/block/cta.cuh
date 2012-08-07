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

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace block {


/**
 *
 */
template <
	typename KernelPolicy,
	typename KeyType,
	typename ValueType>
struct Cta
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Appropriate unsigned-bits representation of KeyType
	typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

	static const UnsignedBits 					MIN_KEY 			= KeyTraits<KeyType>::MIN_KEY;
	static const UnsignedBits 					MAX_KEY 			= KeyTraits<KeyType>::MAX_KEY;
	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= KernelPolicy::LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= KernelPolicy::STORE_MODIFIER;

	enum
	{
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		CTA_THREADS 				= KernelPolicy::CTA_THREADS,
		WARPS						= CTA_THREADS / WARP_THREADS,

		KEYS_PER_THREAD				= KernelPolicy::THREAD_ELEMENTS,
		TILE_ELEMENTS				= KernelPolicy::TILE_ELEMENTS,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,
	};


	// CtaRadixSort utility type
	typedef CtaRadixSort<
		UnsignedBits,
		CTA_THREADS,
		KEYS_PER_THREAD,
		RADIX_BITS,
		ValueType,
		KernelPolicy::SMEM_CONFIG> CtaRadixSort;

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		Partition								partition;

		union
		{
			typename CtaRadixSort::SmemStorage 	sorting_storage;
			UnsignedBits						key_exchange[TILE_ELEMENTS];
		};
	};


	//---------------------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	SmemStorage 		&smem_storage;
	Partition			*d_partitions_out;
	KeyType 			*d_keys_in;
	KeyType 			*d_keys_out;
	KeyType 			*d_keys_final;
	ValueType 			*d_values_in;
	ValueType			*d_values_out;
	ValueType			*d_values_final;
	int					low_bit;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		Partition		*d_partitions_in,
		Partition		*d_partitions_out,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		KeyType 		*d_keys_final,
		util::NullType 	*d_values_in,
		util::NullType 	*d_values_out,
		util::NullType 	*d_values_final,
		int				low_bit) :
			smem_storage(smem_storage),
			d_partitions_out(d_partitions_out),
			d_keys_in(d_keys_in),
			d_keys_out(d_keys_out),
			d_keys_final(d_keys_final),
			d_values_in(d_values_in),
			d_values_out(d_values_out),
			d_values_final(d_values_final),
			low_bit(low_bit)
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
		}

		__syncthreads();
	}


	/**
	 *
	 */
	template <typename T, typename SizeT>
	__device__ __forceinline__ void LoadTile(
		T				*exchange,
		T 				*items,
		T 				*d_in,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int thread_offset = threadIdx.x + (KEY * CTA_THREADS);
			if (thread_offset < guarded_elements)
			{
				items[KEY] = d_in[cta_offset + thread_offset];
			}
		}

		__syncthreads();

		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			exchange[threadIdx.x + (KEY * CTA_THREADS)] = items[KEY];
		}

		__syncthreads();

		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			items[KEY] = exchange[(threadIdx.x * KEYS_PER_THREAD) + KEY];
		}
	}


	/**
	 *
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void StoreTile(
		T 				*items,
		T 				*d_out,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int global_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (global_offset < guarded_elements)
			{
				d_out[cta_offset + global_offset] = items[KEY];
			}
		}
	}


	/**
	 * Block sort
	 */
	__device__ __forceinline__ void BlockSort()
	{
		UnsignedBits keys[KEYS_PER_THREAD];

		// Initialize keys to default key value
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = MAX_KEY;
		}

		// Load keys
		LoadTile(
			smem_storage.key_exchange,
			keys,
			reinterpret_cast<UnsignedBits*>(d_keys_in),
			smem_storage.partition.offset,
			smem_storage.partition.num_elements);

		__syncthreads();

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
		}

		// Sort
		CtaRadixSort::SortThreadToCtaStride(
			smem_storage.sorting_storage,
			keys,
			low_bit,
			smem_storage.partition.current_bit - low_bit);

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleOut(keys[KEY]);
		}

		// Store keys
		StoreTile(
			keys,
			reinterpret_cast<UnsignedBits*>(d_keys_final),
			smem_storage.partition.offset,
			smem_storage.partition.num_elements);
	}


	/**
	 * ProcessTile.  (Specialized for keys-only sorting.)
	 */
	__device__ __forceinline__ void ProcessWorkRange()
	{
		// Quit if there is no work
		if (smem_storage.partition.num_elements == 0) return;

		// Choose block or pass sort
		if (smem_storage.partition.num_elements < TILE_ELEMENTS)
		{
			// Block sort the remainder of the radix bits
			BlockSort();
		}
		else
		{
			// CTA pass
		}
	}

};


} // namespace block
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
