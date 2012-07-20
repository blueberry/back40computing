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

		LOG_CTA_THREADS 			= KernelPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_KEYS_PER_THREAD 		= KernelPolicy::LOG_THREAD_ELEMENTS,
		KEYS_PER_THREAD				= 1 << LOG_KEYS_PER_THREAD,

		LOG_TILE_ELEMENTS			= LOG_CTA_THREADS + LOG_KEYS_PER_THREAD,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		PADDING_ELEMENTS			= TILE_ELEMENTS >> LOG_MEM_BANKS,
	};


	// CtaRadixSort utility type
	typedef CtaRadixSort<
		UnsignedBits,
		LOG_CTA_THREADS,
		LOG_KEYS_PER_THREAD,
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
	// Template iteration
	//---------------------------------------------------------------------

	template <int COUNT, int MAX>
	struct Iterate
	{
		template <typename T>
		static __device__ __forceinline__ void GatherScatterIn(
			T *d_in,
			T *buffer)
		{
			int global_offset 	= (COUNT * CTA_THREADS) + threadIdx.x;
			int shared_offset 	= global_offset;
			shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			T key = buffer[shared_offset] = d_in[global_offset];

			Iterate<COUNT + 1, MAX>::GatherScatterIn(d_in, buffer);
		}

		template <typename T>
		static __device__ __forceinline__ void GatherScatterOut(
			T *d_out,
			T *buffer)
		{
			int global_offset 	= (COUNT * CTA_THREADS) + threadIdx.x;
			int shared_offset 	= global_offset;
			shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			d_out[global_offset] = buffer[shared_offset];

			Iterate<COUNT + 1, MAX>::GatherScatterOut(d_out, buffer);
		}
	};

	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		template <typename T>
		static __device__ __forceinline__ void GatherScatterIn(T*, T*) {}

		template <typename T>
		static __device__ __forceinline__ void GatherScatterOut(T*, T*) {}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * ProcessTile
	 */
	static __device__ __forceinline__ void ProcessTile(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		KeyType 		*d_out_keys,
		ValueType 		*d_in_values,
		ValueType 		*d_out_values,
		unsigned int 	current_bit,
		unsigned int 	bits_remaining,
		unsigned int 	num_elements)
	{
		// Load items into shared memory
		Iterate<0, KEYS_PER_THREAD>::GatherScatterIn(
			reinterpret_cast<UnsignedBits*>(d_in_keys),
			smem_storage.sorting_storage.key_exchange);

		__syncthreads();

		// Sort
		CtaRadixSort::Sort(
			smem_storage.sorting_storage,
			current_bit,
			bits_remaining);

		// Store items from shared memory
		Iterate<0, KEYS_PER_THREAD>::GatherScatterOut(
			reinterpret_cast<UnsignedBits*>(d_out_keys),
			smem_storage.sorting_storage.key_exchange);
	}

};


} // namespace single
} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
