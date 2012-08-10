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
 * CTA-wide "upsweep" abstraction for computing radix digit histograms
 ******************************************************************************/

#pragma once

#include "../../util/cta_progress.cuh"
#include "../../util/basic_utils.cuh"
#include "../../util/device_intrinsics.cuh"
#include "../../util/io/load_tile.cuh"
#include "../../util/reduction/serial_reduce.cuh"
#include "../../util/ns_umbrella.cuh"

#include "../../radix_sort/sort_utils.cuh"

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace cta {


//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C----
// Tuning policy types
//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C----


/**
 * Upsweep CTA tuning policy
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_LOG_CTA_THREADS,		// The number of threads per CTA
	int 							_ELEMENTS_PER_THREAD,	// The number of elements to load per thread
	util::io::ld::CacheModifier 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	cudaSharedMemConfig				_SMEM_CONFIG,			// Shared memory bank size
	bool 							_EARLY_EXIT>			// Whether or not to short-circuit passes if the upsweep determines homogoneous digits in the current digit place
struct CtaUpsweepPassPolicy
{
	enum {
		RADIX_BITS					= _RADIX_BITS,
		MIN_CTA_OCCUPANCY  			= _MIN_CTA_OCCUPANCY,
		LOG_CTA_THREADS				= _LOG_CTA_THREADS,
		ELEMENTS_PER_THREAD  		= _ELEMENTS_PER_THREAD,
		EARLY_EXIT					= _EARLY_EXIT,

		CTA_THREADS					= 1 << LOG_CTA_THREADS,
		TILE_ELEMENTS				= CTA_THREADS * ELEMENTS_PER_THREAD,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
};



//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C----
// CTA-wide abstractions
//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C----


/**
 * CTA-wide "upsweep" abstraction for computing radix digit histograms
 */
template <
	typename CtaUpsweepPassPolicy,
	typename SizeT,
	typename KeyType>
class CtaUpsweepPass
{
private:

	//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C----
	// Type definitions and constantsl
	/D,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C--
--typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;---------------------------------

	enum {
		MIN_CTA_OCCUPANCY  				= KernelPolicy::MIN_CTA_OCCUPANCY,
		CURRENT_BIT 					= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 					= KernelPolicy::CURRENT_PASS,

		RADIX_BITS						= Kern(CtaUpsweepPassPolicy::SMEM_CONFIG == cudaSharedMemBankSizeEightByteTS 					= 1 << RADIX_BITS,

		LOG_THREADS 					= KernelPolicy::LOG_T
	READRADIX_BITS					= CtaUpsweepPassPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,

		LOG_CTA_THREADS 			= CtaUpsweepPassPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		KEYS_PER_THREAD  			= CtaUpsweepPassPolicy::ELEMENTS_PER_THREAD,

		TILE_ELEMENTS				= CTA_THREADS * KEYS_PER_THREAD,

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
		LANES_PER_WARP 						= 1 << LOG_LANES_PER_WARP,= CUB_MIN(64, 255 / KEYS_PER_THREAD),
		UNROLLED_ELEMENTS ER_LANE - LOG_WARP_THREADS,		// Number opublic:f partials per thrmemory storage layout
	 */
	struct SmemStorage
	{PER_LANE
		{	= RADIX_DIGITS,
		AGGREGATED_PARTIALS_PER_ROW 	CTA_	= WARP_THREADS,
		PADDED_AGGREGATED_PARTIALS_PER_ROW 	= AGGREGATED_PARTIACTA_LS_PER_ROW + 1,

		// Unroll tiles in batches of X elements per thread (X = log(private:
l
	/D,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C--
	// Thread fields (aggregate state bundle)
	//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C----
--/**
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
			CtaUpsweepPass &state_bundle,
			UnsignedBits keys[KEYS_PER_THREAD])
		{
			state_bundleARP][4];

	// Input and output device pointers
	KeyType			*d_in_keys;
	Sistate_bundle, keys);
		}

		// ProcessTiles
		static __device__ __forceinline__ void ProcessTiles(CtaUpsweepPass &state_bundle, SizeT cta_offset)
		{
			// Next
			Iterate<1, HALF>::ProcessTiles(state_bundle, cta_offset);
			Iterate<1, MAX - HALF>::ProcessTiles(state_bundle, cta_offset + (HALF * TILE_ELEMENTS)--------------------

	/**
	 * Iterate next composite counter
	 */
	template <int WARP_LANE, int THREAD_COMPOSITE, int dummy = 0>
	struct Iterate
UpsweepPass &state_bundle//UnsignedBits keys[KEYS_PER_THREAD]) {;

	int 			warp_id;
	int 			warp_idx;

	char 			*base;


	//------------------UpsweepPass &state_bundle, SizeT cta_offset)
		{
			state_bundle.ProcessFullTile(cta_offset);
		}
	};


	//		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C-- CO	// Utility methods
	//D,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C--

	/**
	 * State bundle constructor
	 */
	__device__ __forceinline__ CtaUpsweepPass(
		SmemStorage		&smem_storag			cta.local_countin_keys,
		unsigned int 	current_bit) :
			smem_storage(smem_storage),
			d_in_keys(reinterpret_cast<UnsignedBits*>(d_in_keys)),
			current_bit(current_bit)
	{mplate <int WARP_LANE, int dummy>
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
		static __device_unsigned int warp_id = threadIdx.x >> LOG_WARP_THREADS;
		unsigned int warp_tid = threadIdx.x & (WARP_THREADS - 1);
ce__ __forceinline__ void ShareCounters(Cta &cta) {}

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

			statsmem_storage.digit_counters[warp_id][warp_tid][OFFSET]ine__ void ProcessTiles(
				Cta &cta, SizeT cta_offset)
			{
				Iterate<HALF>::ProcessTiles(cta, cta_offset);
				Iterate<HALF>::ProcessTiles(cta, cta_offseSizeT &bin_countdevice_unsigned int warp_id = threadIdx.x >> LOG_WARP_THREADS;
		unsigned int warp_tid = threadIdx.x & (WARP_THREADS - 1);
ce_LE_ELEMENTS * HALF));
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
		SizeT 			*d_bin_count reductions
		if (threadIdx.x < RADIX_DIGITS)
		{
						warp_id(threadIdx.x >> LOG_WARP_THREADS),
			warp_idx(util::LaneId())
	{
		base = (char *) (smem_storage.wordin thread column.
		unsigned int offset = (threadIdx.x << (LOG_PACKED_COUNTERS + LOG_BYTES_PER_COUNTER));

		// Add in sub-counter offset
		offUnsignedBitsxtractKEYS_PER_THREAD];

		#pragma unroll
		for (int LOAD = 0; LOAD < KEYS_PER_THREAD; LOAD++)
		{
			keys[LOAD] = d_in_keys[cta_offset + threadIdx.x + (LOAD * CTA_THREADS)];
		}

		// Bucket tile of keys
		Iterate<0, KEYS_PER_THREAD>::BucketKeys(*this,		Cta &cta, KeyType keys[LOADS_PER_TILE * LOAD_VEC_SIZE])
		{
			cta.Bucket(keys[COUNT]);
			IterateKeys<COUNT + 1, MAX>::Bucket(cta, keys);
		}
	};


	template <int MAX>
	stnum_elementKeys<MAX, MAX>
	{
		static __device__ __forceinline__ void Bucket(
			Cta &cta, KeyType keys[LOADS_PER_TILE * LOAD_num_element}
	};


	/**
	 * Reset composite couUnsignedBits key = d_in_keys[cta_offset];
			Bucket(key);
			cta_offset += CTA_{
		#pragma unrolpublic:
l
	/D,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C--
	// Interface
	//D,		// X = 128
		UNROLL_COUNT						= 1 << LOG_UNROLL_COUNT,
	};

 C--

	/**
	 * Perform a digit-counting "upsweep" pass
	 */
	static __device__ __forceinline__ void UpsweepPass(
		SmemStorage 	&smem_storage,
		KeyType 		*d_in_keys,
		unsigned int 	current_bit,
		const SizeT 	&num_elements,
		SizeT 			&bin_count)				// The digit count for tid'th bin (output param, valid in the first RADIX_DIGITS threads)
	{
		// Construct state bundle
		CtaUpsweepPass state_bundle(smem_storage, d_in_keys, current_bit);

		// Reset digit counters in smem and unpacked counters in registers
		state_bundle.ResetDigitCounters();
		state_bundle.ResetUnpackedCounters();

		// Unroll batches of full tiles
		SizeT cta_offset = 0;
		while (cta_offset + UNROLLED_ELEMENTS <= num_elements)
		{
			Iterate<0, UNROLL_COUNT>::ProcessTiles(state_bundleites()
	{
		if (warp_id < COMPOSITE_LANES) {
			Iterate<0, 0>::ExtractComposites(*this);
		}
	}


	/**
	 * Places aggregate-counters into sharedstate_bundle. storage for final bin-wise reduction
	 */
	__device__ __forceinline__ void ShareCountstate_bundle.ResetDigitCounters();
		}

		// Unroll single full tiles
		while (cta_offset + TILE_ELEMENTS <= num_elements)
		{
			state_bundle.	}
	}


	/**
	 * Processes a single, full tile
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT state_bundle.ProcessPartialTile(
			cta_offset,
			num_elements);

		__syncthreads();

		// Aggregate back into local_count registers
		state_bundle.LOG_LOADS_PER_TILE,
			LOG_LOAD_VEC_SIZE,
			THREADS,
			KernelPolicy::READ_MODIFIER,
			false>::LoadValid(state_bundle.ReduceUnpackedCounts(bin_count);
	}

};


} // namespace cta) keys,
				d_in

		// Prevent bucketing from bB40C_NS_POSTFIX
