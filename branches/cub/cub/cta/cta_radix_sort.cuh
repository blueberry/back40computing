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

/**
 * \file
 * The cub::CtaRadixSort type provides variants of parallel radix sorting of unsigned numeric types across threads within a CTA.
 */


#pragma once

#include "../ns_wrapper.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "cta_exchange.cuh"
#include "cta_radix_rank.cuh"

CUB_NS_PREFIX

/// CUB namespace
namespace cub {


/**
 * \brief The CtaRadixSort type provides variants of parallel radix sorting of unsigned numeric types across threads within a CTA.  ![](sorting_logo.png)
 *
 * \tparam KeyType              Key type
 * \tparam CTA_THREADS          The CTA size in threads
 * \tparam ITEMS_PER_THREAD     The number of items per thread
 * \tparam ValueType            <b>[optional]</b> Value type (default: cub::NullType)
 * \tparam RADIX_BITS           <b>[optional]</b> The number of radix bits per digit place (default: 5 bits)
 * \tparam SMEM_CONFIG          <b>[optional]</b> Shared memory bank mode (default: \p cudaSharedMemBankSizeFourByte)
 *
 * <b>Overview</b>
 * \par
 * The <em>radix sort</em> method relies upon a positional representation for
 * keys, i.e., each key is comprised of an ordered sequence of numeral symbols
 * (i.e., digits) specified from least-significant to most-significant.  For a
 * given input sequence of keys and a set of rules specifying a total ordering
 * of the symbolic alphabet, the radix sorting method produces a lexicographic
 * ordering of those keys.
 *
 * \par
 * The parallel operations exposed by this type assume <em>n</em>-element
 * lists that are partitioned evenly among \p CTA_THREADS threads,
 * with thread<sub><em>i</em></sub> owning the <em>i</em><sup>th</sup>
 * element (or <em>i</em><sup>th</sup> segment of consecutive elements).
 *
 * <b>Features</b>
 * \par
 * - Blah
 * - Blah
 *
 * <b>Algorithm</b>
 * \par
 * These parallel radix sorting variants have <em>O</em>(<em>n</em>) work complexity and are implemented in XXX phases:
 * -# blah
 * -# blah
 *
 * <b>Important Considerations</b>
 * \par
 * - After any CtaRadixSort operation, a subsequent CTA barrier (<tt>__syncthreads()</tt>) is
 *   required if the supplied CtaRadixSort::SmemStorage is to be reused/repurposed by the CTA.
 * - Keys must be in a form suitable for radix ranking (i.e., unsigned integer types).
 * - The operations are most efficient (lowest instruction overhead) when:
 *      - \p CTA_THREADS is a multiple of the architecture's warp size
 *
 * <b>Examples</b>
 * \par
 * - <b>Example 1:</b> Simple radix sort of 32-bit integer keys
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int CTA_THREADS>
 *      __global__ void SomeKernel(...)
 *      {
 *
 *      \endcode
 *
 * \par
 * - <b>Example 2:</b> Simple key-value radix sort of 32-bit integer keys paird with 32-bit float values
 *      \code
 *      #include <cub.cuh>
 *
 *      template <int CTA_THREADS>
 *      __global__ void SomeKernel(...)
 *      {
 *
 *      \endcode
 */
template <
	typename				KeyType,
	int 					CTA_THREADS,
	int						ITEMS_PER_THREAD,
	typename 				ValueType 		= NullType,
	int 					RADIX_BITS 		= 5,
	cudaSharedMemConfig 	SMEM_CONFIG 	= cudaSharedMemBankSizeFourByte>
class CtaRadixSort
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	/// CtaRadixRank utility type
	typedef CtaRadixRank<
		CTA_THREADS,
		RADIX_BITS,
		SMEM_CONFIG> CtaRadixRank;

	/// CtaExchange utility type for keys
	typedef CtaExchange<
		KeyType,
		CTA_THREADS,
		ITEMS_PER_THREAD> KeyCtaExchange;

	/// CtaExchange utility type for values
	typedef CtaExchange<
		ValueType,
		CTA_THREADS,
		ITEMS_PER_THREAD> ValueCtaExchange;

    /// Shared memory storage layout type
    struct SmemStorage
    {
        union
        {
            typename CtaRadixRank::SmemStorage          ranking_storage;
            typename KeyCtaExchange::SmemStorage        key_xchg_storage;
            typename ValueCtaExchange::SmemStorage      value_xchg_storage;
        };
    };

public:

    /// The operations exposed by CtaRadixSort require shared memory of this
    /// type.  This opaque storage can be allocated directly using the
    /// <tt>__shared__</tt> keyword.  Alternatively, it can be aliased to
    /// externally allocated shared memory or <tt>union</tt>'d with other types
    /// to facilitate shared memory reuse.
    typedef SmemStorage SmemStorage;


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
		SmemStorage			&smem_storage,								///< [in] Shared reference to opaque SmemStorage layout
		KeyType 			keys[ITEMS_PER_THREAD],						///< [in-out] Keys to sort
		unsigned int 		current_bit = 0,							///< [in] <b>[optional]</b> The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		///< [in] <b>[optional]</b> The number of bits needed for key comparison
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
		SmemStorage			&smem_storage,								///< [in] Shared reference to opaque SmemStorage layout
		KeyType 			keys[ITEMS_PER_THREAD],						///< [in-out] Keys to sort
		unsigned int 		current_bit = 0,							///< [in] <b>[optional]</b> The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		///< [in] <b>[optional]</b> The number of bits needed for key comparison
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
		SmemStorage			&smem_storage,								///< [in] Shared reference to opaque SmemStorage layout
		KeyType 			keys[ITEMS_PER_THREAD],						///< [in-out] Keys to sort
		unsigned int 		current_bit = 0,							///< [in] <b>[optional]</b> The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		///< [in] <b>[optional]</b> The number of bits needed for key comparison
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
		SmemStorage			&smem_storage,								///< [in] Shared reference to opaque SmemStorage layout
		KeyType 			keys[ITEMS_PER_THREAD],						///< [in-out] Keys to sort
		ValueType 			values[ITEMS_PER_THREAD],					///< [in-out] Values to sort
		unsigned int 		current_bit = 0,							///< [in] <b>[optional]</b> The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		///< [in] <b>[optional]</b> The number of bits needed for key comparison
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
		SmemStorage			&smem_storage,								///< [in] Shared reference to opaque SmemStorage layout
		KeyType 			keys[ITEMS_PER_THREAD],						///< [in-out] Keys to sort
		ValueType 			values[ITEMS_PER_THREAD],					///< [in-out] Values to sort
		unsigned int 		current_bit = 0,							///< [in] <b>[optional]</b> The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		///< [in] <b>[optional]</b> The number of bits needed for key comparison
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
		SmemStorage			&smem_storage,								///< [in] Shared reference to opaque SmemStorage layout
		KeyType 			keys[ITEMS_PER_THREAD],						///< [in-out] Keys to sort
		ValueType 			values[ITEMS_PER_THREAD],					///< [in-out] Values to sort
		unsigned int 		current_bit = 0,							///< [in] <b>[optional]</b> The least-significant bit needed for key comparison
		const unsigned int	&bits_remaining = sizeof(KeyType) * 8)		///< [in] <b>[optional]</b> The number of bits needed for key comparison
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
