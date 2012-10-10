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
 * CTA abstractions for commonplace all-to-all exchanges between threads
 ******************************************************************************/

#pragma once

#include "../ns_wrapper.cuh"
#include "../device_props.cuh"
#include "../ptx_intrinsics.cuh"
#include "../type_utils.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Cooperative all-to-all exchange abstractions for CTAs.
 */
template <
	typename 	T,						/// The data type to be exchanged
	int 		CTA_THREADS,			/// The CTA size in threads
	int			ITEMS_PER_THREAD>		/// The number of items per thread
class CtaExchange
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

	enum
	{
		TILE_ITEMS					= CTA_THREADS * ITEMS_PER_THREAD,

		LOG_SMEM_BANKS				= DeviceProps::LOG_SMEM_BANKS,
		SMEM_BANKS					= 1 << LOG_SMEM_BANKS,

		// Insert padding if the number of items per thread is a power of two
		PADDING  					= ((ITEMS_PER_THREAD & (ITEMS_PER_THREAD - 1)) == 0),
		PADDING_ELEMENTS			= (PADDING) ? (TILE_ITEMS >> LOG_SMEM_BANKS) : 0,
	};

public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		T exchange[TILE_ITEMS + PADDING_ELEMENTS];
	};


private:

	static __device__ __forceinline__ void ScatterBlocked(
		T items[ITEMS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
			if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
			buffer[item_offset] = items[ITEM];
		}
	}

	static __device__ __forceinline__ void ScatterStriped(
		T items[ITEMS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
			if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
			buffer[item_offset] = items[ITEM];
		}
	}

	static __device__ __forceinline__ void GatherBlocked(
		T items[ITEMS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (threadIdx.x * ITEMS_PER_THREAD) + ITEM;
			if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
			items[ITEM] = buffer[item_offset];
		}
	}

	static __device__ __forceinline__ void GatherStriped(
		T items[ITEMS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = (ITEM * CTA_THREADS) + threadIdx.x;
			if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
			items[ITEM] = buffer[item_offset];
		}
	}

	static __device__ __forceinline__ void ScatterRanked(
		T 				items[ITEMS_PER_THREAD],
		unsigned int 	ranks[ITEMS_PER_THREAD],
		T 				*buffer)
	{
		#pragma unroll
		for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
		{
			int item_offset = ranks[ITEM];
			if (PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);

			buffer[item_offset] = items[ITEM];
		}
	}


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

public:

	/**
	 * Transposes data items across the CTA from "blocked" arrangement
	 * to CTA-striped arrangement.
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
	static __device__ __forceinline__ void TransposeBlockedStriped(
		SmemStorage			&smem_storage,					/// (opaque) Shared memory storage
		T 					items[ITEMS_PER_THREAD])		/// (in/out) Items to exchange
	{
		// Scatter items to shared memory
		ScatterStriped(items, smem_storage.exchange);

		__syncthreads();

		// Gather items from shared memory
		GatherBlocked(items, smem_storage.exchange);
	}


	/**
	 * Transposes data items across the CTA from "CTA-striped"
	 * arrangement
	 * to blocked arrangement.
	 *
	 * As input, the aggregate set of items is assumed to be ordered
	 * across threads in "CTA-striped" fashion, i.e., each thread owns
	 * an array of items having logical stride CTA_THREADS between each item
	 * (e.g., items[0] in thread-0 is logically followed by items[0] in
	 * thread-1, and so on).
	 *
	 * As output, the aggregate set of items is assumed to be ordered across
	 * threads in "blocked fashion", i.e., each thread owns an array of
	 * logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	static __device__ __forceinline__ void TransposeStripedBlocked(
		SmemStorage			&smem_storage,					/// (opaque) Shared memory storage
		T 					items[ITEMS_PER_THREAD])		/// (in/out) Items to exchange
	{
		// Scatter items to shared memory
		ScatterBlocked(items, smem_storage.exchange);

		__syncthreads();

		// Gather items from shared memory
		GatherStriped(items, smem_storage.exchange);
	}


	/**
	 * Rearranges data items by rank across the CTA in "blocked"
	 * arrangement.
	 *
	 * As output, the aggregate set of items is assumed to be ordered across
	 * threads in "blocked fashion", i.e., each thread owns an array of
	 * logically-consecutive items (and consecutive thread ranks own
	 * logically-consecutive arrays).
	 */
	static __device__ __forceinline__ void ScatterGatherBlocked(
		SmemStorage			&smem_storage,					/// (opaque) Shared memory storage
		T 					items[ITEMS_PER_THREAD],		/// (in/out) Items to exchange
		unsigned int 		ranks[ITEMS_PER_THREAD])		/// (in) Corresponding scatter ranks
	{
		// Scatter items to shared memory
		Scatter(items, ranks, smem_storage.exchange);

		__syncthreads();

		// Gather items from shared memory
		GatherBlocked(items, smem_storage.exchange);
	}


	/**
	 * Rearranges data items by rank across the CTA in "CTA-striped"
	 * arrangement.
	 *
	 * As output, the aggregate set of items is assumed to be ordered
	 * across threads in "CTA-striped" fashion, i.e., each thread owns an
	 * array of items having logical stride CTA_THREADS between each item
	 * (e.g., items[0] in thread-0 is logically followed by items[0] in
	 * thread-1, and so on).
	 */
	static __device__ __forceinline__ void ScatterGatherStriped(
		SmemStorage			&smem_storage,					/// (opaque) Shared memory storage
		T 					items[ITEMS_PER_THREAD],		/// (in/out) Items to exchange
		unsigned int 		ranks[ITEMS_PER_THREAD])		/// (in) Corresponding scatter ranks
	{
		// Scatter items to shared memory
		Scatter(items, ranks, smem_storage.exchange);

		__syncthreads();

		// Gather items from shared memory
		GatherStriped(items, smem_storage.exchange);
	}



};

} // namespace cub
CUB_NS_POSTFIX
