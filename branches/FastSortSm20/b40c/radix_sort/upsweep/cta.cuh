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
 * CTA-processing functionality for radix sort upsweep reduction kernels
 ******************************************************************************/

#pragma once

#include "../../radix_sort/sort_utils.cuh"
#include "../../util/basic_utils.cuh"
#include "../../util/device_intrinsics.cuh"
#include "../../util/io/load_tile.cuh"
#include "../../util/reduction/serial_reduce.cuh"
#include "../../util/ns_umbrella.cuh"

B40C_NS_PREFIXinclude <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace upsweep {

/**
 * Radix sort upsweep reduction CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
	typename KeyType>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------typedef typename KeyTraits<KeyType>::UnsignedBits 	UnsignedBits;---------------------------------

	enum {
		MIN_CTA_OCCUPANCY  				= KernelPolicy::MIN_CTA_OCCUPANCY,
		CURRENT_BIT 					= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 					= KernelPolicy::CURRENT_PASS,

		RADIX_BITS						= Kern(KernelPolicy::SMEM_CONFIG == cudaSharedMemBankSizeEightByteTS 					= 1 << RADIX_BITS,

		LOG_THREADS 					= KernelPolicy::LOG_THREADRADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		LOG_CTA_THREADS 			= KernelPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_LOAD_VEC_SIZE  			= KernelPolicy::LOG_LOAD_VEC_SIZE,
		LOAD_VEC_SIZE				= 1 << LOG_LOAD_VEC_SIZE,

		LOG_LOADS_PER_TILE 			= KernelPolicy::LOG_LOADS_PER_TILE,
		LOADS_PER_TILE				= 1 << LOG_LOADS_PER_TILE,

		LOG_THREAD_ELEMENTS			= LOG_LOAD_VEC_SIZE + LOG_LOADS_PER_TILE,
		THREAD_ELEMENTS				= 1 << LOG_THREAD_ELEMENTS,

		LOG_TILE_ELEMENTS 			= LOG_THREAD_ELEMENTS + LOG_CTA_THREADS,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		BYTES_PER_COUNTER			= sizeof(DigitCounter),
		LOG_BYTES_PER_COUNTER		= util::Log2<BYTES_PER_COUNTER>::VALUE,

		PACKING_RATIO				= sizeof(PackedCounter) / sizeof(DigitCounter),
		LOG_PACKING_RATIO			= util::Log2<PACKING_RATIO>::VALUE,

		LOG_COUNTER_LANES 			= CUB_MAX(0, RADIX_BITS - LOG_PACKING_RATIO),
		COUNTER_LANES prevent bin-counter overflow, we must partially-aggregate the
		// 8-bit composite counters back into SizeT-bit registers periodically.  Each lane
		// is assigned to a warp for aggregation.  Each lane is therefore equivalent to
		// four = CUB_MAX(0, LOG_COUNTER_LANES - LOG_WARPS),
		LANES_PER_WARP ES_PER_WARP					= CUB_MAX(0, LOG_COMPOSITE_LANES - LOG_WARPS),
		LANES_PER_WARP 						= 1 << LOG_LANES_PER_WARP,= CUB_MIN(64, 255 / THREAD_ELEMENTS),
		UNROLLED_ELEMENTS ER_LANE - LOG_WARP_THREADS,		// Number of partials per thread to aggregate
		COMPOSITES_PER_LANE_PER_THREAD 		= 1 << LOG_COMPOSITES_PER_LANE_PER_THREAD,

		AGGREGATED_ROWS						= RADIX_DIGITS,
		AGGREGATED_PARTIALS_PER_ROW 	CTA_	= WARP_THREADS,
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIACTA_LS_PER_ROW + 1,

		// Unroll tiles in batches of X elements per thread (X = log(255) is maximum without risking overflow)
		LOG_UNROLL_COUNT 					= 6 - LOG_TILE_ELEMENTS_PER_THREAD,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};



	/**
	 * Shared storage for radix distributionn sorting upsweep
	 */
	struct SmemStorage
	{
		union {
			// Composite counter storage
			union {
				cha	local_counts[LANES_PER_WARP][PACKING_RATIO];

	// Input and output device pointers
	UnsignedBits		*d_in_keys;
	SizeT				*d_spine;

	// Warp-id and lane-id
	int 				warp_id;
	int 				warp_tid;

	DigitCounter		*base_counter;

	// The least-significant bit position of the current digit to extract
	unsigned int 		current_bit;



	//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C

----------------------------------------------------------
	// Fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStoraenum {
			HALF = (MAX / 2),
		};

		// BucketKeys
		static __device__ __forceinline__ void BucketKeys(
			Cta &cta,
			UnsignedBits keys[THREAD_ELEMENTSs[LANES_PER_WARP][4];

	// Input and output device pointers
	KeyType			*d_in_keys;
	SizeT			*d_spine;

	int 			warp_id;
	int 			warp_idx;

	char 			*base;


	//--------------------------------------------------// Next
			Iterate<1, HALF>::ProcessTiles(cta, cta_offset);
			Iterate<1, MAX - HALF>::ProcessTiles(cta, cta_offset + (HALF * TILE_ELEMENTS)--------------------

	/**
	 * Iterate next composite counter
	 */
	template <int WARP_LANE, int THREAD_COMPOSITE, int dummy = 0>
	struct Iterate
	{
		//UnsignedBits keys[THREAD_ELEMENTS]) {;

	int 			warp_id;
	int 			warp_idx;

	char 			*base;


	//------------------------------------------
		{
			cta.ProcessFullTile(cta_offset);
		}
	};


	//EAD,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 CO	// Methods
	//EAD,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 COMPOSITE_OFFSET + 0);
			cta.local_counts[WARP_LANE][1] += *(cta.base + LANE_OFFSET + COMPOSITE_OFFSET + 1);
			cta.local_countin_keys,
		unsigned int 	current_bit) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			warp_id(threadIdx.x >> LOG_WARP_THREADS),
			warp_tid(util::LaneId()),
			current_bit(current_bit>::ExtractComposites(cta);
		}
	};

	/**
	 * Iterate next lane
	 */
	template <int WARP_LANE, int dummy>
	struct Iterate<WARP_LANE, COMPOSITES_PER_LANE_PER_THREAD, dummy>
	{
		// ExtractComUnsignedBits
		static __dePerform transform op
		UnsignedBits converted_key = KeyTraits<KeyType>::TwiddleIn(key);

		// Add in sub-counter offset
		UnsignedBits sub_counter = util::BFE(converted_key, current_bit, LOG_PACKING_RATIO);

		// Add in row offset
		UnsignedBits row_offset = util::BFE(converted_key, current_bit + LOG_PACKING_RATIO, LOG_COUNTER_LANES);

		// Increment counter
		smem_storage.digit_counters[row_offset][threadIdx.x][sub_counter]++;
 COMPOSITES_PER_LANE_PER_THREAD>::ShareCounters(cta);
		}

		// ResetCounters
		static __device__ __forceinline__ void ResetCounters(Cta &cta)
		{
			cta.local_counts[WARP_LANE][0] = 0;
			cta.local_counts[WARP_LANE][1] = 0;
			cta.local_counts[WARP_LANE][2] = 0;
			cta.local_counts[WARP_LANE][3] = 0;

			Iterate<WARP_LANE + 1, COMPOSITES_PER_LANE_PER_THREAD>::ResetCounters(cta);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LANES_PER_WARP, 0, dummy>
	{
		// ExtractComposites
		static __device__ __forceinline__ void ExtractComposites(Cta &cta) {}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct Iterate<LANES_PER_WARP, COMPOSITES_PER_LANE_PER_THREAD, dummy>
	{
		// ShareCounters
		static __device__ __forceinline__ void ShareCounters(Cta &cta) {}

		// ResetCounters
		static __device__ __forceinline__ void ResetCounters(Cta &cta) {}
	};


	//---------------------------------------------------------------------
	// HCTA_elper structure for tile unrolling
	//---------------------------------------------------------------------

	/**
	 * Unrolled tile processing
	 */
	struct UnrollTiles
	{
		// Recurse over counts
		template CTA_<int UNROLL_COUNT, int __dummy = 0>
		struct Iterate
		{
			static const int HALF = UNROLL_COUNT / 2;

			static __device__ __forceinline__ void ProcessTiles(
				Cta &cta, SizeT cta_offset)
			{
				Iterate<HALF>::ProcessTiles(cta, cta_offset);
				Iterate<HALF>::ProcessTiles(cta, cta_offset + (TILE_ELEMENTS * HALF));
			}
		};

		// Terminate (process one tile)
		template <int __dummy>
		struct Iterate<1, __dummy>
		{
			static __device__ __forceinline__ void ProcessTiles(
				Cta &cta, SizeT cta_offset)
			{
				cta.ProcessFullTile(cta_offset);
			}
		};
	};





	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(d_in_keys),
			d_spine(d_spine),
			warp_id(threadIdx.x >> LOG_WARP_THREADS),
			warp_idx(util::LaneId())
	{
		base = (char *) (smem_storage.words[warp_id] + warp_idx);
	}


	/**
	 * Bucket a key into smem counters
	 */
	__device__ __forceinline__ void BSTORt(KeyType key)
	{
		// Compute byte offset of smem counter.  Add in thread column.
		unsigned int offset = (threadIdx.x << (LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER));

		// Add in sub-counter offset
		offUnsignedBitsxtract<
			KeyType,
			CURRENT_BIT,
			LOG_PACKED_COUNTERS,
			LOG_BYTES_PER_COUNTER>::SuperBFE(
				key,
				offset);

		// Add in CTA_THREADS,
			KernelPolicy::LOAD_MODIFIER,
			false>::LoadValid(
				E_LANES,
			LOG_THREADS + (LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER)>::SuperBFE(
				key,
				offset);

		((unsigned char *) smem_storage.counters)[offset]++;


	}


	template <int COUNT, int MAX>
	struct IterateKeysHREAD_ELEMENTS>::BucketKeys(*this, (UnsignedBits
			Cta &cta, KeyType keys[LOADS_PER_TILE * LOAD_VEC_SIZE])
		{
			cta.Bucket(keys[COUNT]);
			IterateKeys<COUNT + 1, MAX>::Bucket(cta, keys);
		}
	};


	template <int MAX>
	struct IterateKeys<MAX, MAX>
	{
		static __device__ __forceinline__ void Bucket(
			Cta &cta, KeyType keys[LOADS_PER_TILE * LOAD_VEC_SIZE]) {}
	};


	/**
	 * Reset composite couUnsignedBits key = d_in_keys[cta_offset];
			Bucket(key);
			cta_offset += CTA_{
		#pragma unroll
		for (int LANE = 0; LANE < COMPOSITE_LANES; ++LANE) {
			smem_storage.words[LANE][threadIdx.x] = 0;
		}
	}


	/**
	 * Resets the aggregate counters
	 */
	__device__ __forceinline__ void ResetCounters()
	{
		Iterate<0, COMPOSITES_PER_LANE_PER_THREAD>::ResetCounters(*this);
	}


	/**
	 * Extracts and aggregates the shared-memory composite counters for each
	 * composite-counter lane owned by this warp
	 */
	__device__ __forceinline__ void ExtractComposites()
	{
		if (warp_id < COMPOSITE_LANES) {
			Iterate<0, 0>::ExtractComposites(*this);
		}
	}


	/**
	 * Places aggregate-counters into shared storage for final bin-wise reduction
	 */
	__device__ __forceinline__ void ShareCounters()
	{
		if (warp_id < COMPOSITE_LANES) {
			Iterate<0, COMPOSITES_PER_LANE_PER_THREAD>::ShareCounters(*this);
		}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{
		// Tile of keys
		KeyType keys[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Read tile of keys
		util::io::LoadTile<
			LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(
				(KeyType (*)[LOAD_VEC_SIZE]) keys,
				d_in_keys,
				cta_offset);

		// Prevent bucketing from bB40C_NS_POSTFIX
