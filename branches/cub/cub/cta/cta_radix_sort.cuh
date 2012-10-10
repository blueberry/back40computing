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
 * CTA abstraction for sorting keys (unsigned bits) by radix digit.
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "cta_exchange.cuh"
#include "cta_radix_rank.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * CTA collective abstraction for sorting keys (unsigned bits) by radix digit.
 *
 * Keys must be in a form suitable for radix ranking (i.e., unsigned integer types).
 */
template <
	typename				KeyType,											/// Key type
	int 					CTA_THREADS,										/// The CTA size in threads
	int						ITEMS_PER_THREAD,									/// The number of items per thread
	typename 				ValueType 		= NullType,							/// (optional) Value type
	int 					RADIX_BITS 		= 5,								/// (optional) The number of radix bits per digit place
	cudaSharedMemConfig 	SMEM_CONFIG 	= cudaSharedMemBankSizeFourByte>	/// (optional) Shared memory bank size
class CtaRadixSort
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	// CtaRadixRank utility type
	typedef CtaRadixRank<
		CTA_THREADS,
		RADIX_BITS,
		SMEM_CONFIG> CtaRadixRank;

	// CtaExchange utility type for keys
	typedef CtaExchange<
		KeyType,
		CTA_THREADS,
		ITEMS_PER_THREAD> KeyCtaExchange;

	// CtaExchange utility type for values
	typedef CtaExchange<
		ValueType,
		CTA_THREADS,
		ITEMS_PER_THREAD> ValueCtaExchange;

public:

	/**
	 * Opaque shared memory storage layout
	 */
	struct SmemStorage
	{
		union
		{
			typename CtaRadixRank::SmemStorage			ranking_storage;
			typename KeyCtaExchange::SmemStorage		key_xchg_storage;
			typename ValueCtaExchange::SmemStorage		value_xchg_storage;
		};
	};


	//---------------------------------------------------------------------
	// Keys-only interface
	//---------------------------------------------------------------------

public:

	/**
	 * Keys-only, least-significant-digit (LSD) radix sorting, "blocked"
	 * arrangement.
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	static __device__ __forceinline__ void SortBlocked(
		SmemStorage			&smem_storage,								/// (opaque) Shared memory storage
		KeyType 			keys[ITEMS_PER_THREAD],						/// (in/out) Keys to sort
		unsigned int 		current_bit = 0,							/// (in, optional) The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		/// (in, optional) The number of bits needed for key comparison
	{
		// Radix sorting passes
		while (true)
		{
			// Rank the keys within the CTA
			unsigned int ranks[ITEMS_PER_THREAD];

			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			current_bit += RADIX_BITS;

			__syncthreads();

			// Exchange keys through shared memory in "blocked" arrangement
			KeyCtaExchange::ScatterGatherBlocked(
				smem_storage.key_xchg_storage,
				keys,
				ranks);

			// Check if done
			if (current_bit >= bits_remaining)
			{
				break;
			}

			__syncthreads();
		}
	}


	/**
	 * Keys-only, least-significant-digit (LSD) radix sorting, "CTA-striped"
	 * arrangement.
	 *
	 * The aggregate set of items is assumed to be ordered
	 * across threads in "CTA-striped" fashion, i.e., each thread owns
	 * an array of items having logical stride CTA_THREADS between each item
	 * (e.g., items[0] in thread-0 is logically followed by items[0] in
	 * thread-1, and so on).
	 */
	static __device__ __forceinline__ void SortStriped(
		SmemStorage			&smem_storage,								/// (opaque) Shared memory storage
		KeyType 			keys[ITEMS_PER_THREAD],						/// (in/out) Keys to sort
		unsigned int 		current_bit = 0,							/// (in, optional) The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		/// (in, optional) The number of bits needed for key comparison
	{
		// Transpose keys from "CTA-striped" to "blocked" arrangement
		KeyCtaExchange::TransposeStripedBlocked(
			smem_storage.key_xchg_storage,
			keys);

		__syncthreads();

		// Radix sorting passes
		while (true)
		{
			// Rank the keys within the CTA
			unsigned int ranks[ITEMS_PER_THREAD];

			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			current_bit += RADIX_BITS;

			__syncthreads();

			// Check if done
			if (current_bit >= bits_remaining)
			{
				// Exchange keys through shared memory in "striped" arrangement
				KeyCtaExchange::ScatterGatherStriped(
					smem_storage.key_xchg_storage,
					keys,
					ranks);

				break;
			}

			// Exchange keys through shared memory in "blocked" arrangement
			KeyCtaExchange::ScatterGatherBlocked(
				smem_storage.key_xchg_storage,
				keys,
				ranks);

			__syncthreads();
		}
	}


	/**
	 * Keys-only, least-significant-digit (LSD) radix sorting, "blocked"
	 * arrangement (input) --> "CTA-striped" arrangement (output).
	 *
	 * As input, the aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 *
	 * As output, the aggregate set of items is assumed to be ordered
	 * across threads in "CTA-striped" fashion, i.e., each thread owns an
	 * array of items having logical stride CTA_THREADS between each item
	 * (e.g., items[0] in thread-0 is logically followed by items[0] in
	 * thread-1, and so on).
	 */
	static __device__ __forceinline__ void SortBlockedToStriped(
		SmemStorage			&smem_storage,								/// (opaque) Shared memory storage
		KeyType 			keys[ITEMS_PER_THREAD],						/// (in/out) Keys to sort
		unsigned int 		current_bit = 0,							/// (in, optional) The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		/// (in, optional) The number of bits needed for key comparison
	{
		// Radix sorting passes
		while (true)
		{
			// Rank the keys within the CTA
			unsigned int ranks[ITEMS_PER_THREAD];

			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			current_bit += RADIX_BITS;

			__syncthreads();

			// Check if done
			if (current_bit >= bits_remaining)
			{
				// Exchange keys through shared memory in "striped" arrangement
				KeyCtaExchange::ScatterGatherStriped(
					smem_storage.key_xchg_storage,
					keys,
					ranks);

				break;
			}

			// Exchange keys through shared memory in "blocked" arrangement
			KeyCtaExchange::ScatterGatherBlocked(
				smem_storage.key_xchg_storage,
				keys,
				ranks);

			__syncthreads();
		}
	}


	//---------------------------------------------------------------------
	// Key-value pairs interface
	//---------------------------------------------------------------------

public:

	/**
	 * Key-value pairs, least-significant-digit (LSD) radix sorting, "blocked"
	 * arrangement.
	 *
	 * The aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	static __device__ __forceinline__ void SortBlocked(
		SmemStorage			&smem_storage,								/// (opaque) Shared memory storage
		KeyType 			keys[ITEMS_PER_THREAD],						/// (in/out) Keys to sort
		ValueType 			values[ITEMS_PER_THREAD],					/// (in/out) Values to sort
		unsigned int 		current_bit = 0,							/// (in, optional) The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		/// (in, optional) The number of bits needed for key comparison
	{
		// Radix sorting passes
		while (true)
		{
			// Rank the keys within the CTA
			unsigned int ranks[ITEMS_PER_THREAD];

			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			current_bit += RADIX_BITS;

			__syncthreads();

			// Exchange keys through shared memory in "blocked" arrangement
			KeyCtaExchange::ScatterGatherBlocked(
				smem_storage.key_xchg_storage,
				keys,
				ranks);

			__syncthreads();

			// Exchange values through shared memory in "blocked" arrangement
			ValueCtaExchange::ScatterGatherBlocked(
				smem_storage.value_xchg_storage,
				values,
				ranks);

			// Check if done
			if (current_bit >= bits_remaining)
			{
				break;
			}

			__syncthreads();
		}
	}


	/**
	 * Key-value pairs, least-significant-digit (LSD) radix sorting, "CTA-striped"
	 * arrangement.
	 *
	 * The aggregate set of items is assumed to be ordered across threads in
	 * "CTA-striped" fashion, i.e., each thread owns an array of items having
	 * logical stride CTA_THREADS between each item (e.g., items[0] in
	 * thread-0 is logically followed by items[0] in thread-1, and so on).
	 */
	static __device__ __forceinline__ void SortStriped(
		SmemStorage			&smem_storage,								/// (opaque) Shared memory storage
		KeyType 			keys[ITEMS_PER_THREAD],						/// (in/out) Keys to sort
		ValueType 			values[ITEMS_PER_THREAD],					/// (in/out) Values to sort
		unsigned int 		current_bit = 0,							/// (in, optional) The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		/// (in, optional) The number of bits needed for key comparison
	{
		// Transpose keys from "CTA-striped" to "blocked" arrangement
		KeyCtaExchange::TransposeStripedBlocked(
			smem_storage.key_xchg_storage,
			keys);

		__syncthreads();

		// Transpose values from "CTA-striped" to "blocked" arrangement
		KeyCtaExchange::TransposeStripedBlocked(
			smem_storage.key_xchg_storage,
			keys);

		__syncthreads();

		// Radix sorting passes
		while (true)
		{
			// Rank the keys within the CTA
			unsigned int ranks[ITEMS_PER_THREAD];

			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			current_bit += RADIX_BITS;

			__syncthreads();

			// Check if done
			if (current_bit >= bits_remaining)
			{
				// Exchange keys through shared memory in "striped" arrangement
				KeyCtaExchange::ScatterGatherStriped(
					smem_storage.key_xchg_storage,
					keys,
					ranks);

				__syncthreads();

				// Exchange values through shared memory in "striped" arrangement
				ValueCtaExchange::ScatterGatherStriped(
					smem_storage.value_xchg_storage,
					values,
					ranks);

				break;
			}

			// Exchange keys through shared memory in "blocked" arrangement
			KeyCtaExchange::ScatterGatherBlocked(
				smem_storage.key_xchg_storage,
				keys,
				ranks);

			__syncthreads();

			// Exchange values through shared memory in "blocked" arrangement
			ValueCtaExchange::ScatterGatherBlocked(
				smem_storage.value_xchg_storage,
				values,
				ranks);

			__syncthreads();
		}
	}


	/**
	 * Key-value pairs, least-significant-digit (LSD) radix sorting, "blocked"
	 * arrangement (input) --> "CTA-striped" arrangement (output).
	 *
	 * As input, the aggregate set of items is assumed to be ordered across
	 * threads in "blocked" fashion, i.e., each thread owns an array
	 * of logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 *
	 * As output, the aggregate set of items is assumed to be ordered
	 * across threads in "CTA-striped" fashion, i.e., each thread owns an
	 * array of items having logical stride CTA_THREADS between each item
	 * (e.g., items[0] in thread-0 is logically followed by items[0] in
	 * thread-1, and so on).
	 */
	static __device__ __forceinline__ void SortBlockedToStriped(
		SmemStorage			&smem_storage,								/// (opaque) Shared memory storage
		KeyType 			keys[ITEMS_PER_THREAD],						/// (in/out) Keys to sort
		ValueType 			values[ITEMS_PER_THREAD],					/// (in/out) Values to sort
		unsigned int 		current_bit = 0,							/// (in, optional) The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		/// (in, optional) The number of bits needed for key comparison
	{
		// Radix sorting passes
		while (true)
		{
			// Rank the keys within the CTA
			unsigned int ranks[ITEMS_PER_THREAD];

			CtaRadixRank::RankKeys(
				smem_storage.ranking_storage,
				keys,
				ranks,
				current_bit);

			current_bit += RADIX_BITS;

			__syncthreads();

			// Check if done
			if (current_bit >= bits_remaining)
			{
				// Exchange keys through shared memory in "striped" arrangement
				KeyCtaExchange::ScatterGatherStriped(
					smem_storage.key_xchg_storage,
					keys,
					ranks);

				__syncthreads();

				// Exchange values through shared memory in "striped" arrangement
				ValueCtaExchange::ScatterGatherStriped(
					smem_storage.value_xchg_storage,
					values,
					ranks);

				break;
			}

			// Exchange keys through shared memory in "blocked" arrangement
			KeyCtaExchange::ScatterGatherBlocked(
				smem_storage.key_xchg_storage,
				keys,
				ranks);

			__syncthreads();

			// Exchange values through shared memory in "blocked" arrangement
			ValueCtaExchange::ScatterGatherBlocked(
				smem_storage.value_xchg_storage,
				values,
				ranks);

			__syncthreads();
		}
	}

};




} // namespace cub
CUB_NS_POSTFIX
