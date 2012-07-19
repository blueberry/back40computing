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
	// Utility methods
	//---------------------------------------------------------------------

	/**
	 * Scatter ranked items to shared memory buffer
	 */
	template <typename T>
	static __device__ __forceinline__ void ScatterRanked(
		unsigned int 	ranks[KEYS_PER_THREAD],
		T 				items[KEYS_PER_THREAD],
		T 				*buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int offset = ranks[KEY];

			// Workaround for (CUAD4.2+NVCC+abi+m64) bug when sorting 16-bit key-value pairs
			offset = (sizeof(ValueType) == 2) ?
				(offset >> LOG_MEM_BANKS) + offset :
				util::SHR_ADD(offset, LOG_MEM_BANKS, offset);

			buffer[offset] = items[KEY];
		}
	}


	/**
	 * Gather items from shared memory buffer
	 */
	template <typename T>
	static __device__ __forceinline__ void GatherShared(
		T items[KEYS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int gather_offset = (util::SHR_ADD(threadIdx.x, LOG_MEM_BANKS, threadIdx.x) +
				(KEY * CTA_THREADS) + ((KEY * CTA_THREADS) >> LOG_MEM_BANKS));

			items[KEY] = buffer[gather_offset];
		}
	}


	/**
	 * Exchange ranked items through shared memory
	 */
	template <typename T>
	static __device__ __forceinline__ void ScatterGather(
		T 				items[KEYS_PER_THREAD],
		T 				*buffer,
		unsigned int 	ranks[KEYS_PER_THREAD])
	{
		// Scatter to shared memory first (for better write-coalescing during global scatter)
		ScatterRanked(ranks, items, buffer);

		__syncthreads();

		// Gather sorted keys from shared memory
		GatherShared(items, buffer);

		__syncthreads();
	}


	/**
	 * Exchange ranked items through shared memory (specialized for key-only sorting)
	 */
	static __device__ __forceinline__ void ScatterGather(
		util::NullType	items[KEYS_PER_THREAD],
		util::NullType	*buffer,
		unsigned int 	ranks[KEYS_PER_THREAD])
	{}

public:


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Key-value sorting
	 */
	static __device__ __forceinline__ void Sort(
		SmemStorage		&smem_storage,									// Shared memory storage
		UnsignedBits 	keys[KEYS_PER_THREAD],							// Calling thread's keys
		ValueType 		values[KEYS_PER_THREAD],						// Calling thread's values
		unsigned int 	current_bit = 0,								// The least-significant bit needed for key comparison
		unsigned int	bits_remaining = sizeof(UnsignedBits) * 8)		// The number of bits needed for key comparison
	{
		// Radix sorting passes
		while (current_bit < bits_remaining)
		{
			// Rank the keys within the CTA
			unsigned int ranks[KEYS_PER_THREAD];
			CtaRadixRank::RankKeys(smem_storage.ranking_storage, keys, ranks, current_bit);

			__syncthreads();

			// Exchange keys
			ScatterGather(keys, smem_storage.key_exchange, ranks);

			// Exchange values
			ScatterGather(values, smem_storage.value_exchange, ranks);

			current_bit += RADIX_BITS;
		}
	}


	/**
	 * Keys-only sorting
	 */
	static __device__ __forceinline__ void Sort(
		SmemStorage		&smem_storage,									// Shared memory storage
		UnsignedBits 	keys[KEYS_PER_THREAD],							// Calling thread's keys
		unsigned int 	current_bit = 0,								// The least-significant bit needed for key comparison
		unsigned int	bits_remaining = sizeof(UnsignedBits) * 8)		// The number of bits needed for key comparison
	{
		// Radix sorting passes
		while (current_bit < bits_remaining)
		{
			// Rank the keys within the CTA
			unsigned int ranks[KEYS_PER_THREAD];
			CtaRadixRank::RankKeys(smem_storage.ranking_storage, keys, ranks, current_bit);

			__syncthreads();

			// Exchange keys
			ScatterGather(keys, smem_storage.key_exchange, ranks);

			current_bit += RADIX_BITS;
		}
	}
};




} // namespace radix_sort
} // namespace b40c
B40C_NS_POSTFIX
