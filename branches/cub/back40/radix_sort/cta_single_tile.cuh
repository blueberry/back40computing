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
 * CTA-wide abstraction for sorting a single tile of input
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta/cta_radix_sort.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace cta {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------

/**
 * Single tile CTA tuning policy
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_CTA_THREADS,			// The number of threads per CTA
	int 							_THREAD_ELEMENTS,		// The number of consecutive keys to process per thread per tile
	util::io::ld::CacheModifier	 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaSingleTilePolicy
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



//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------

/**
 * CTA-wide abstraction for sorting a single tile of input
 */
template <
	typename CtaSingleTilePolicy,
	typename KeyType,
	typename ValueType>
class CtaSingleTile
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Appropriate unsigned-bits representation of KeyType
	typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

	static const UnsignedBits 					MIN_KEY 			= KeyTraits<KeyType>::MIN_KEY;
	static const UnsignedBits 					MAX_KEY 			= KeyTraits<KeyType>::MAX_KEY;
	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= CtaSingleTilePolicy::LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= CtaSingleTilePolicy::STORE_MODIFIER;

	enum
	{
		RADIX_BITS					= CtaSingleTilePolicy::RADIX_BITS,
		KEYS_PER_THREAD				= CtaSingleTilePolicy::THREAD_ELEMENTS,
		TILE_ELEMENTS				= CtaSingleTilePolicy::TILE_ELEMENTS,
	};


	// CtaRadixSort utility type
	typedef CtaRadixSort<
		UnsignedBits,
		CTA_THREADS,
		KEYS_PER_THREAD,
		RADIX_BITS,
		ValueType,
		CtaSingleTilePolicy::SMEM_CONFIG> CtaRadixSort;

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
		const int 		&num_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int thread_offset = threadIdx.x + (KEY * CTA_THREADS);
			if (thread_offset < num_elements)
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
		const int 		&num_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int global_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (global_offset < num_elements)
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
		SmemStorage 	&cta_smem_storage,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		util::NullType 	*d_values_in,
		util::NullType 	*d_values_out,
		unsigned int 	current_bit,
		unsigned int 	bits_remaining,
		const int 		&num_elements)
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
			cta_smem_storage.key_exchange,
			keys,
			reinterpret_cast<UnsignedBits*>(d_keys_in),
			cta_offset,
			num_elements);

		__syncthreads();

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
		}

		// Sort
		CtaRadixSort::SortThreadToCtaStride(
			cta_smem_storage.sorting_storage,
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
			num_elements);
	}

};


} // namespace cta
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
