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
 * CTA-processing functionality for single-CTA radix sort kernel
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
namespace single {


/**
 * Partitioning downsweep scan CTA
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

		// Insert padding if the number of keys per thread is a power of two
		PADDING  					= ((KEYS_PER_THREAD & (KEYS_PER_THREAD - 1)) == 0),
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
		union
		{
			typename CtaRadixSort::SmemStorage sorting_storage;
		};
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * ProcessTile
	 */
	template <typename SizeT>
	static __device__ __forceinline__ void ProcessTile(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys,
		ValueType 		*d_values,
		unsigned int 	current_bit,
		unsigned int 	bits_remaining,
		SizeT			cta_offset,
		int 			guarded_elements)
	{
		UnsignedBits keys[KEYS_PER_THREAD];

		// Initialize keys
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = MAX_KEY;;
		}

		// Load keys
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int global_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (global_offset < guarded_elements)
			{
				keys[KEY] = d_keys[cta_offset + global_offset];
			}
		}

		__syncthreads();

		// Store keys
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int shared_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (PADDING) shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			smem_storage.sorting_storage.key_exchange[shared_offset] = keys[KEY];
		}

		__syncthreads();

		// Sort
		CtaRadixSort::Sort(
			smem_storage.sorting_storage,
			current_bit,
			bits_remaining);

		// Load keys
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int shared_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (PADDING) shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			keys[KEY] = smem_storage.sorting_storage.key_exchange[shared_offset];
		}

		// Store keys
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int global_offset = (KEY * CTA_THREADS) + threadIdx.x;
			if (global_offset < guarded_elements)
			{
				d_keys[cta_offset + global_offset] = keys[KEY];;
			}
		}
	}

};


} // namespace single
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
