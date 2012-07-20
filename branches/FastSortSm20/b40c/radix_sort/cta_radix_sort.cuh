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
 * CTA collective abstraction for sorting keys (unsigned bits) by radix digit.
 ******************************************************************************/

#pragma once

#include "../radix_sort/sort_utils.cuh"

#include "../util/basic_utils.cuh"
#include "../util/reduction/serial_reduce.cuh"
#include "../util/scan/serial_scan.cuh"
#include "../util/ns_umbrella.cuh"


B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {


/**
 * CTA collective abstraction for sorting keys (unsigned bits) by radix digit.
 *
 * Keys must be in a form suitable for radix ranking (i.e., unsigned bits).
 */
template <
	typename				UnsignedBits,
	int 					LOG_CTA_THREADS,
	int						LOG_KEYS_PER_THREAD,
	int 					RADIX_BITS,
	typename 				ValueType = util::NullType,
	cudaSharedMemConfig 	SMEM_CONFIG = cudaSharedMemBankSizeFourByte>	// Shared memory bank size
class CtaRadixSort
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	enum
	{
		CTA_THREADS					= 1 << LOG_CTA_THREADS,
		KEYS_PER_THREAD				= 1 << LOG_KEYS_PER_THREAD,

		LOG_TILE_ELEMENTS			= LOG_CTA_THREADS + LOG_KEYS_PER_THREAD,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		PADDING_ELEMENTS			= TILE_ELEMENTS >> LOG_MEM_BANKS
	};

	// CtaRadixRank utility type
	typedef CtaRadixRank<
		LOG_CTA_THREADS,
		RADIX_BITS,
		SMEM_CONFIG> CtaRadixRank;

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaRadixRank::SmemStorage	ranking_storage;
			UnsignedBits						key_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
			ValueType 							value_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
		};
	};

private:

	//---------------------------------------------------------------------
	// Template iteration
	//---------------------------------------------------------------------

	template <int COUNT, int MAX>
	struct Iterate
	{
		template <typename T>
		static __device__ __forceinline__ void Gather(
			T items[KEYS_PER_THREAD],
			T *buffer)
		{
			int shared_offset = (threadIdx.x * KEYS_PER_THREAD) + COUNT;
			shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			items[COUNT] = buffer[shared_offset];

			Iterate<COUNT + 1, MAX>::Gather(items, buffer);
		}

		template <typename T>
		static __device__ __forceinline__ void Scatter(
			unsigned int 	ranks[KEYS_PER_THREAD],
			T 				items[KEYS_PER_THREAD],
			T 				*buffer)
		{
			int shared_offset = ranks[COUNT];
			shared_offset = util::SHR_ADD(shared_offset, LOG_MEM_BANKS, shared_offset);

			buffer[shared_offset] = items[COUNT];

			Iterate<COUNT + 1, MAX>::Scatter(ranks, items, buffer);
		}
	};

	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		template <typename T>
		static __device__ __forceinline__ void Gather(
			T items[KEYS_PER_THREAD],
			T *buffer) {}

		template <typename T>
		static __device__ __forceinline__ void Scatter(
			unsigned int 	ranks[KEYS_PER_THREAD],
			T 				items[KEYS_PER_THREAD],
			T 				*buffer) {}
	};


public:


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Keys-only sorting
	 */
	static __device__ __forceinline__ void Sort(
		SmemStorage		&smem_storage,									// Shared memory storage
		unsigned int 	current_bit = 0,								// The least-significant bit needed for key comparison
		unsigned int	bits_remaining = sizeof(UnsignedBits) * 8)		// The number of bits needed for key comparison
	{
		// Radix sorting passes

		for (
			unsigned int bit = current_bit;
			bit < bits_remaining;
			bit += RADIX_BITS)
		{
			UnsignedBits keys[KEYS_PER_THREAD];
			unsigned int ranks[KEYS_PER_THREAD];

			// Gather keys from shared memory
			Iterate<0, KEYS_PER_THREAD>::Gather(keys, smem_storage.key_exchange);

			__syncthreads();

			// Rank the keys within the CTA
			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				bit);

			__syncthreads();

			// Scatter keys to shared memory
			Iterate<0, KEYS_PER_THREAD>::Scatter(ranks, keys, smem_storage.key_exchange);

			__syncthreads();
		}
	}
};




} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
