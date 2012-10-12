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

#include "../../cub/cub.cuh"
#include "../ns_wrapper.cuh"
#include "sort_utils.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------

/**
 * Single tile CTA tuning policy
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_CTA_THREADS,			// The number of threads per CTA
	int 							_THREAD_ITEMS,			// The number of consecutive items to load per thread per tile
	cub::PtxLoadModifier 				_LOAD_MODIFIER,			// Load cache-modifier
	cub::PtxStoreModifier				_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG>			// Shared memory bank size
struct CtaSingleTilePolicy
{
	enum
	{
		RADIX_BITS					= _RADIX_BITS,
		CTA_THREADS 				= _CTA_THREADS,
		THREAD_ITEMS 				= _THREAD_ITEMS,
		TILE_ITEMS					= CTA_THREADS * THREAD_ITEMS,
	};

	static const cub::PtxLoadModifier 		LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const cub::PtxStoreModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig	SMEM_CONFIG			= _SMEM_CONFIG;
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

	static const UnsignedBits 	MIN_KEY 	= KeyTraits<KeyType>::MIN_KEY;
	static const UnsignedBits 	MAX_KEY 	= KeyTraits<KeyType>::MAX_KEY;

	enum
	{
		CTA_THREADS					= CtaSingleTilePolicy::CTA_THREADS,
		RADIX_BITS					= CtaSingleTilePolicy::RADIX_BITS,
		KEYS_PER_THREAD				= CtaSingleTilePolicy::THREAD_ITEMS,
		TILE_ITEMS				= CtaSingleTilePolicy::TILE_ITEMS,
	};


	// CtaRadixSort utility type
	typedef cub::CtaRadixSort<
		UnsignedBits,
		CTA_THREADS,
		KEYS_PER_THREAD,
		RADIX_BITS,
		ValueType,
		CtaSingleTilePolicy::SMEM_CONFIG> CtaRadixSortT;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaRadixSortT::SmemStorage 	sorting_storage;
			UnsignedBits							key_exchange[TILE_ITEMS];
		};
	};


private:

	//---------------------------------------------------------------------
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 *
	 */
	template <typename T>
	static __device__ __forceinline__ void LoadTile(
		T				*exchange,
		T 				*items,
		T 				*d_in,
		const int 		&num_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int thread_offset = threadIdx.x + (KEY * CTA_THREADS);
			if (thread_offset < num_elements)
			{
				items[KEY] = d_in[thread_offset];
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
	template <typename T>
	static __device__ __forceinline__ void StoreTile(
		T 				*items,
		T 				*d_out,
		const int 		&num_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int thread_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (thread_offset < num_elements)
			{
				d_out[thread_offset] = items[KEY];
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
	static __device__ __forceinline__ void Sort(
		SmemStorage 		&smem_storage,
		KeyType 			*d_keys_in,
		KeyType 			*d_keys_out,
		cub::NullType 		*d_values_in,
		cub::NullType 		*d_values_out,
		unsigned int 		current_bit,
		const unsigned int 	&bits_remaining,
		const int 			&num_elements)
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
			num_elements);

		__syncthreads();

		// Twiddle key bits if necessary
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = KeyTraits<KeyType>::TwiddleIn(keys[KEY]);
		}

		// Sort
		CtaRadixSortT::SortThreadToCtaStride(
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
			num_elements);
	}

};


} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
