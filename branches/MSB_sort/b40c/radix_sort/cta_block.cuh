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
 * "Block-sort" CTA abstraction for sorting small tiles of input
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


/**
 * Block CTA tuning policy
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_CTA_THREADS,			// The number of threads per CTA
	int 							_THREAD_ELEMENTS,		// The number of consecutive keys to process per thread per tile
	util::io::ld::CacheModifier	 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaBlockPolicy
{
	enum
	{
		RADIX_BITS					= _RADIX_BITS,
		CTA_THREADS 				= _CTA_THREADS,
		THREAD_ELEMENTS 			= _THREAD_ELEMENTS,

		TILE_ELEMENTS				= CTA_THREADS * THREAD_ELEMENTS,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
};



/**
 * "Block-sort" CTA abstraction for sorting small tiles of input
 */
template <
	typename CtaBlockPolicy,
	typename KeyType,
	typename ValueType>
class CtaBlock
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Appropriate unsigned-bits representation of KeyType
	typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

	static const UnsignedBits 					MIN_KEY 			= KeyTraits<KeyType>::MIN_KEY;
	static const UnsignedBits 					MAX_KEY 			= KeyTraits<KeyType>::MAX_KEY;
	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= CtaBlockPolicy::LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= CtaBlockPolicy::STORE_MODIFIER;

	enum
	{
		RADIX_BITS					= CtaBlockPolicy::RADIX_BITS,
		KEYS_PER_THREAD				= CtaBlockPolicy::THREAD_ELEMENTS,
		TILE_ELEMENTS				= CtaBlockPolicy::TILE_ELEMENTS,
	};


	// CtaRadixSort utility type
	typedef CtaRadixSort<
		UnsignedBits,
		CTA_THREADS,
		KEYS_PER_THREAD,
		RADIX_BITS,
		ValueType,
		CtaBlockPolicy::SMEM_CONFIG> CtaRadixSort;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaRadixSort::SmemStorage 	sorting_storage;
			UnsignedBits						key_exchange[TILE_ELEMENTS];
		};
	};


private:

	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 *
	 */
	template <typename T, typename SizeT>
	static __device__ __forceinline__ void LoadTile(
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

public:

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * ProcessTile.  (Specialized for keys-only sorting.)
	 */
	template <typename SizeT>
	static __device__ __forceinline__ void Sort(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		util::NullType 	*d_values_in,
		util::NullType 	*d_values_out,
		unsigned int 	current_bit,
		unsigned int 	bits_remaining,
		const SizeT		&cta_offset,
		const int 		&guarded_elements)
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
			cta_offset,
			guarded_elements);

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
			current_bit,
			bits_remaining);

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleOut(keys[KEY]);
		}

		// Store keys
		StoreTile(
			keys,
			reinterpret_cast<UnsignedBits*>(d_keys_out),
			cta_offset,
			guarded_elements);
	}

};


/**
 * Kernel entry point
 */
template <
	typename CtaBlockPolicy,
	typename KeyType,
	typename ValueType>
__launch_bounds__ (CtaBlockPolicy::CTA_THREADS, 1)
__global__
void Kernel(
	KeyType 							*d_keys,
	ValueType 							*d_values,
	unsigned int 						current_bit,
	unsigned int						bits_remaining,
	unsigned int 						num_elements)
{
	// CTA abstraction type
	typedef CtaBlock<CtaBlockPolicy, KeyType, ValueType> CtaBlock;

	// Shared memory pool
	__shared__ typename CtaBlock::SmemStorage smem_storage;

	CtaBlock::ProcessTile(
		smem_storage,
		d_keys,
		d_keys,
		d_values,
		d_values,
		current_bit,
		bits_remaining,
		int(0),
		num_elements);
}


} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
