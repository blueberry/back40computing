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
 * CTA-wide "downsweep" abstraction for distributing keys from
 * a range of input tiles.
 ******************************************************************************/

#pragma once

#include "../../util/basic_utils.cuh"
#include "../../util/device_intrinsics.cuh"
#include "../../util/io/modified_load.cuh"
#include "../../util/io/modified_store.cuh"
#include "../../util/ns_wrapper.cuh"

#include "../../radix_sort/sort_utils.cuh"
#include "../../radix_sort/cta/cta_radix_rank.cuh"

BACK40_NS_PREFIX
namespace back40 {
namespace radix_sort {
namespace cta {


//---------------------------------------------------------------------
// Tuning policy types
//---------------------------------------------------------------------

/**
 * Types of scattering strategies
 */
enum ScatterStrategy
{
	SCATTER_DIRECT = 0,			// Scatter directly from registers to global bins
	SCATTER_TWO_PHASE,			// First scatter from registers into shared memory bins, then into global bins
	SCATTER_WARP_TWO_PHASE,		// Similar to SCATTER_TWO_PHASE, but with the additional constraint that each warp only perform segment-aligned global writes
};


/**
 * Downsweep CTA tuning policy
 */
template <
	int 							_RADIX_BITS,			// The number of radix bits, i.e., log2(bins)
	int 							_MIN_CTA_OCCUPANCY,		// The minimum CTA occupancy requested for this kernel per SM
	int 							_LOG_CTA_THREADS,		// The number of threads per CTA
	int 							_THREAD_ELEMENTS,		// The number of consecutive keys to process per thread per tile
	util::io::ld::CacheModifier	 	_LOAD_MODIFIER,			// Load cache-modifier
	util::io::st::CacheModifier 	_STORE_MODIFIER,		// Store cache-modifier
	ScatterStrategy 				_SCATTER_STRATEGY,		// Scattering strategy
	cudaSharedMemConfig				_SMEM_CONFIG,			// Shared memory bank size
	bool						 	_EARLY_EXIT>			// Whether or not to short-circuit passes if the upsweep determines homogoneous digits in the current digit place
struct CtaDownsweepPolicy
{
	enum
	{
		RADIX_BITS					= _RADIX_BITS,
		MIN_CTA_OCCUPANCY  			= _MIN_CTA_OCCUPANCY,
		LOG_CTA_THREADS 			= _LOG_CTA_THREADS,
		THREAD_ELEMENTS 			= _THREAD_ELEMENTS,
		EARLY_EXIT					= _EARLY_EXIT,

		CTA_THREADS					= 1 << LOG_CTA_THREADS,
		TILE_ELEMENTS				= CTA_THREADS * THREAD_ELEMENTS,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= _LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= _STORE_MODIFIER;
	static const cudaSharedMemConfig			SMEM_CONFIG			= _SMEM_CONFIG;
	static const ScatterStrategy 				SCATTER_STRATEGY 	= _SCATTER_STRATEGY;
};



//---------------------------------------------------------------------
// CTA-wide abstractions
//---------------------------------------------------------------------


/**
 * CTA-wide "downsweep" abstraction for distributing keys from
 * a range of input tiles.
 */
template <
	typename CtaDownsweepPolicy,
	typename SizeT,
	typename KeyType,
	typename ValueType>
class CtaDownsweep
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	// Appropriate unsigned-bits representation of KeyType
	typedef typename KeyTraits<KeyType>::UnsignedBits UnsignedBits;

	static const UnsignedBits 					MIN_KEY 			= KeyTraits<KeyType>::MIN_KEY;
	static const UnsignedBits 					MAX_KEY 			= KeyTraits<KeyType>::MAX_KEY;
	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= CtaDownsweepPolicy::LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= CtaDownsweepPolicy::STORE_MODIFIER;
	static const ScatterStrategy 				SCATTER_STRATEGY 	= CtaDownsweepPolicy::SCATTER_STRATEGY;

	enum {
		RADIX_BITS					= CtaDownsweepPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,
		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,

		LOG_CTA_THREADS 			= CtaDownsweepPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		KEYS_PER_THREAD 			= CtaDownsweepPolicy::THREAD_ELEMENTS,
		TILE_ELEMENTS				= CtaDownsweepPolicy::TILE_ELEMENTS,

		BYTES_PER_SIZET				= sizeof(SizeT),
		LOG_BYTES_PER_SIZET			= util::Log2<BYTES_PER_SIZET>::VALUE,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		// Whether or not to insert padding for exchanging keys. (Padding is
		// worse than bank conflicts on GPUs that need two-phase scattering)
		PADDED_EXCHANGE 			= false, //(SCATTER_STRATEGY != SCATTER_WARP_TWO_PHASE),
		PADDING_ELEMENTS			= (PADDED_EXCHANGE) ? (TILE_ELEMENTS >> LOG_MEM_BANKS) : 0,

		DIGITS_PER_SCATTER_PASS 	= CTA_THREADS / MEM_BANKS,
		SCATTER_PASSES 				= RADIX_DIGITS / DIGITS_PER_SCATTER_PASS,

		LOG_STORE_TXN_THREADS 		= LOG_MEM_BANKS,
		STORE_TXN_THREADS 			= 1 << LOG_STORE_TXN_THREADS,
	};

	// CtaRadixRank utility type
	typedef CtaRadixRank<
		CTA_THREADS,
		RADIX_BITS,
		CtaDownsweepPolicy::SMEM_CONFIG> CtaRadixRank;


public:

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		SizeT										cta_offset;
		SizeT										cta_offset_limit;
		unsigned int 								digit_prefixes[RADIX_DIGITS + 1];
		SizeT 										digit_offsets[RADIX_DIGITS];

		union
		{
			typename CtaRadixRank::SmemStorage		ranking_storage;
			UnsignedBits							key_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
			ValueType 								value_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
		};
	};


private:

	//---------------------------------------------------------------------
	// Thread fields (aggregate state bundle)
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 				&cta_smem_storage;

	// Input and output device pointers
	UnsignedBits 				*d_keys_in;
	UnsignedBits				*d_keys_out;
	ValueType 					*d_values_in;
	ValueType 					*d_values_out;

	// The global scatter base offset for each digit (valid in the first RADIX_DIGITS threads)
	SizeT 						bin_prefix;

	// The least-significant bit position of the current digit to extract
	unsigned int 				current_bit;


	//---------------------------------------------------------------------
	// Helper structure for templated iteration.  (NVCC currently won't
	// unroll loops with "unexpected control flow".)
	//---------------------------------------------------------------------

	/**
	 * Iterate
	 */
	template <int COUNT, int MAX>
	struct Iterate
	{
		/**
		 * Scatter items to global memory
		 */
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			T 				items[KEYS_PER_THREAD],
			SizeT			digit_offsets[KEYS_PER_THREAD],
			T 				*d_out,
			SizeT 			guarded_elements)
		{
			// Scatter if not out-of-bounds
			int tile_element = threadIdx.x + (COUNT * CTA_THREADS);
			T* scatter = d_out + threadIdx.x + (COUNT * CTA_THREADS) + digit_offsets[COUNT];

			if (FULL_TILE || (tile_element < guarded_elements))
			{
				util::io::ModifiedStore<STORE_MODIFIER>::St(items[COUNT], scatter);
			}

			// Iterate next element
			Iterate<COUNT + 1, MAX>::template ScatterGlobal<FULL_TILE>(
				items, digit_offsets, d_out, guarded_elements);
		}


		/**
		 * Scatter items to global memory
		 */
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			T 				items[KEYS_PER_THREAD],
			unsigned int 	ranks[KEYS_PER_THREAD],
			SizeT			digit_offsets[KEYS_PER_THREAD],
			T 				*d_out,
			SizeT 			guarded_elements)
		{
			// Scatter if not out-of-bounds
			T* scatter = d_out + ranks[COUNT] + digit_offsets[COUNT];

			if (FULL_TILE || (ranks[COUNT] < guarded_elements))
			{
				util::io::ModifiedStore<STORE_MODIFIER>::St(items[COUNT], scatter);
			}

			// Iterate next element
			Iterate<COUNT + 1, MAX>::template ScatterGlobal<FULL_TILE>(
				items, ranks, digit_offsets, d_out, guarded_elements);
		}


		/**
		 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
		 * coalescing rules
		 */
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterPass(
			SmemStorage 	&cta_smem_storage,
			T 				*buffer,
			T 				*d_out,
			SizeT 			valid_elements)
		{
			int store_txn_idx 		= threadIdx.x & (STORE_TXN_THREADS - 1);
			int store_txn_digit 	= threadIdx.x >> LOG_STORE_TXN_THREADS;
			int my_digit 			= (COUNT * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

			if (my_digit < RADIX_DIGITS)
			{
				int my_exclusive_scan 	= cta_smem_storage.digit_prefixes[my_digit];
				int my_inclusive_scan 	= cta_smem_storage.digit_prefixes[my_digit + 1];
				int my_carry 			= cta_smem_storage.digit_offsets[my_digit] + my_exclusive_scan;
				int my_aligned_offset 	= store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

				int gather_offset;
				while ((gather_offset = my_aligned_offset + my_exclusive_scan) < my_inclusive_scan)
				{
					if ((my_aligned_offset >= 0) && (gather_offset < valid_elements))
					{
						int padded_gather_offset = (PADDED_EXCHANGE) ?
							gather_offset = util::SHR_ADD(gather_offset, LOG_MEM_BANKS, gather_offset) :
							gather_offset;

						T datum = buffer[padded_gather_offset];
						d_out[my_carry + my_aligned_offset] = datum;
					}
					my_aligned_offset += STORE_TXN_THREADS;
				}
			}

			// Next scatter pass
			Iterate<COUNT + 1, MAX>::AlignedScatterPass(cta_smem_storage, buffer, d_out, valid_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// ScatterGlobal
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(T[KEYS_PER_THREAD], SizeT[KEYS_PER_THREAD], T*, SizeT) {}

		// ScatterGlobal
		template <bool FULL_TILE, typename T>
		static __device__ __forceinline__ void ScatterGlobal(T[KEYS_PER_THREAD], unsigned int[KEYS_PER_THREAD], SizeT[KEYS_PER_THREAD], T*, SizeT) {}

		// AlignedScatterPass
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterPass(SmemStorage&, T*, T*, SizeT) {}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaDownsweep(
		SmemStorage 	&cta_smem_storage,
		SizeT 			bin_prefix,
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		ValueType 		*d_values_in,
		ValueType 		*d_values_out,
		unsigned int 	current_bit) :
			cta_smem_storage(cta_smem_storage),
			bin_prefix(bin_prefix),
			d_keys_in(reinterpret_cast<UnsignedBits*>(d_keys_in)),
			d_keys_out(reinterpret_cast<UnsignedBits*>(d_keys_out)),
			d_values_in(d_values_in),
			d_values_out(d_values_out),
			current_bit(current_bit)
	{}


	/**
	 * Perform a bit-wise twiddling transformation on keys
	 */
	template <UnsignedBits TwiddleOp(UnsignedBits)>
	__device__ __forceinline__ void TwiddleKeys(
		UnsignedBits 	keys[KEYS_PER_THREAD],
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD])		// out parameter
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			twiddled_keys[KEY] = TwiddleOp(keys[KEY]);
		}
	}


	/**
	 * Scatter ranked items to shared memory buffer
	 */
	template <typename T>
	__device__ __forceinline__ void ScatterRanked(
		unsigned int 	ranks[KEYS_PER_THREAD],
		T 				items[KEYS_PER_THREAD],
		T 				*buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int offset = ranks[KEY];

			if (PADDED_EXCHANGE)
			{
				// Workaround for (CUAD4.2+NVCC+abi+m64) bug when sorting 16-bit key-value pairs
				offset = (sizeof(ValueType) == 2) ?
					(offset >> LOG_MEM_BANKS) + offset :
					util::SHR_ADD(offset, LOG_MEM_BANKS, offset);
			}

			buffer[offset] = items[KEY];
		}
	}


	/**
	 * Gather items from shared memory buffer
	 */
	template <typename T>
	__device__ __forceinline__ void GatherShared(
		T items[KEYS_PER_THREAD],
		T *buffer)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int gather_offset = (PADDED_EXCHANGE) ?
				(util::SHR_ADD(threadIdx.x, LOG_MEM_BANKS, threadIdx.x) +
					(KEY * CTA_THREADS) +
					((KEY * CTA_THREADS) >> LOG_MEM_BANKS)) :
				(threadIdx.x + (KEY * CTA_THREADS));

			items[KEY] = buffer[gather_offset];
		}
	}


	/**
	 * Decodes given keys to lookup digit offsets in shared memory
	 */
	__device__ __forceinline__ void DecodeDigitOffsets(
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD])
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			// Decode address of bin-offset in smem
			UnsignedBits digit = util::BFE(twiddled_keys[KEY], current_bit, RADIX_BITS);

			// Lookup base digit offset from shared memory
			digit_offsets[KEY] = cta_smem_storage.digit_offsets[digit];
		}
	}

	/**
	 * Load tile of keys from global memory
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void LoadKeys(
		UnsignedBits 	keys[KEYS_PER_THREAD],
		SizeT 			cta_offset,
		const SizeT 	&guarded_elements)
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = MAX_KEY;
		}

		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			int thread_offset = threadIdx.x + (KEY * CTA_THREADS);

			if (FULL_TILE || (thread_offset < guarded_elements))
			{
				keys[KEY] = d_keys_in[cta_offset + thread_offset];
			}
		}

		__syncthreads();

		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			cta_smem_storage.key_exchange[threadIdx.x + (KEY * CTA_THREADS)] = keys[KEY];
		}

		__syncthreads();

		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			keys[KEY] = cta_smem_storage.key_exchange[(threadIdx.x * KEYS_PER_THREAD) + KEY];
		}

		__syncthreads();

	}


	/**
	 * Load tile of values from global memory
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void LoadValues(
		ValueType 		values[KEYS_PER_THREAD],
		SizeT 			cta_offset,
		const SizeT 	&guarded_elements)
	{
	}


	/**
	 * Scatter ranked keys to global memory
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ScatterKeys(
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],		// (out parameter)
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			guarded_elements)
	{
		if (SCATTER_STRATEGY == SCATTER_DIRECT)
		{
			// Scatter keys directly to global memory

			// Compute scatter offsets
			DecodeDigitOffsets(twiddled_keys, digit_offsets);

			// Untwiddle keys before outputting
			UnsignedBits keys[KEYS_PER_THREAD];
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(twiddled_keys, keys);

			// Scatter to global
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				keys,
				ranks,
				digit_offsets,
				d_keys_out,
				guarded_elements);
		}
		else if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE)
		{
			// Use warp-aligned scattering of sorted keys from shared memory

			// Untwiddle keys before outputting
			UnsignedBits keys[KEYS_PER_THREAD];
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(twiddled_keys, keys);

			// Scatter to shared memory first
			ScatterRanked(ranks, keys, cta_smem_storage.key_exchange);

			__syncthreads();

			// Gather sorted keys from smem and scatter to global using warp-aligned scattering
			Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
				cta_smem_storage,
				cta_smem_storage.key_exchange,
				d_keys_out,
				guarded_elements);
		}
		else
		{
			// Normal two-phase scatter: exchange through shared memory, then
			// scatter sorted keys to global

			// Scatter to shared memory first (for better write-coalescing during global scatter)
			ScatterRanked(ranks, twiddled_keys, cta_smem_storage.key_exchange);

			__syncthreads();

			// Gather sorted keys from shared memory
			GatherShared(twiddled_keys, cta_smem_storage.key_exchange);

			// Compute scatter offsets
			DecodeDigitOffsets(twiddled_keys, digit_offsets);

			// Untwiddle keys before outputting
			UnsignedBits keys[KEYS_PER_THREAD];
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(twiddled_keys, keys);

			// Scatter keys to global memory
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				keys,
				digit_offsets,
				d_keys_out,
				guarded_elements);
		}
	}


	/**
	 * Truck along associated values
	 */
	template <bool FULL_TILE, typename _ValueType>
	__device__ __forceinline__ void GatherScatterValues(
		_ValueType 		values[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			cta_offset,
		SizeT 			guarded_elements)
	{
		// Load tile of values
		LoadValues<FULL_TILE>(values, cta_offset, guarded_elements);

		if (SCATTER_STRATEGY == SCATTER_DIRECT)
		{
			// Scatter values directly to global memory
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				values,
				ranks,
				digit_offsets,
				d_values_out,
				guarded_elements);
		}
		else if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE)
		{
			__syncthreads();

			// Exchange values through shared memory for better write-coalescing
			ScatterRanked(ranks, values, cta_smem_storage.value_exchange);

			__syncthreads();

			// Use explicitly warp-aligned scattering of values from shared memory
			Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
				cta_smem_storage,
				cta_smem_storage.value_exchange,
				d_values_out,
				guarded_elements);
		}
		else
		{
			__syncthreads();

			// Exchange values through shared memory for better write-coalescing
			ScatterRanked(ranks, values, cta_smem_storage.value_exchange);

			__syncthreads();

			// Gather values from shared
			GatherShared(values, cta_smem_storage.value_exchange);

			// Scatter to global memory
			Iterate<0, KEYS_PER_THREAD>::template ScatterGlobal<FULL_TILE>(
				values,
				digit_offsets,
				d_values_out,
				guarded_elements);
		}
	}


	/**
	 * Truck along associated values (specialized for key-only sorting)
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void GatherScatterValues(
		util::NullType	values[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			cta_offset,
		SizeT 			guarded_elements)
	{}


	/**
	 * Process tile
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = TILE_ELEMENTS)
	{
		// Per-thread tile data
		UnsignedBits 	keys[KEYS_PER_THREAD];					// Keys
		UnsignedBits 	twiddled_keys[KEYS_PER_THREAD];			// Twiddled (if necessary) keys
		ValueType 		values[KEYS_PER_THREAD];				// Values
		unsigned int	ranks[KEYS_PER_THREAD];					// For each key, the local rank within the CTA
		SizeT 			digit_offsets[KEYS_PER_THREAD];			// For each key, the global scatter base offset of the corresponding digit

		// Load tile of keys and twiddle bits if necessary
		LoadKeys<FULL_TILE>(keys, cta_offset, guarded_elements);

		__syncthreads();

		// Twiddle keys
		TwiddleKeys<KeyTraits<KeyType>::TwiddleIn>(keys, twiddled_keys);

		// Rank the twiddled keys
		CtaRadixRank::RankKeys(
			cta_smem_storage.ranking_storage,
			twiddled_keys,
			ranks,
			cta_smem_storage.digit_prefixes,
			current_bit);

		__syncthreads();

		// Update global scatter base offsets for each digit
		if ((CTA_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
		{
			bin_prefix -= cta_smem_storage.digit_prefixes[threadIdx.x];
			cta_smem_storage.digit_offsets[threadIdx.x] = bin_prefix;
			bin_prefix += cta_smem_storage.digit_prefixes[threadIdx.x + 1];
		}

		__syncthreads();

		// Scatter keys
		ScatterKeys<FULL_TILE>(twiddled_keys, digit_offsets, ranks, guarded_elements);

		// Gather/scatter values
		GatherScatterValues<FULL_TILE>(values, digit_offsets, ranks, cta_offset, guarded_elements);
	}


public:

	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Distribute keys from a range of input tiles.
	 */
	static __device__ __forceinline__ void Downsweep(
		SmemStorage 	&cta_smem_storage,
		SizeT 			bin_prefix, 			// The global scatter base offset for each digit (valid in the first RADIX_DIGITS threads)
		KeyType 		*d_keys_in,
		KeyType 		*d_keys_out,
		ValueType 		*d_values_in,
		ValueType 		*d_values_out,
		unsigned int 	current_bit,
		const SizeT 	&num_elements)			// Number of elements for this CTA to process
	{
		// Construct state bundle
		CtaUpsweep cta(
			cta_smem_storage,
			bin_prefix,
			d_keys_in,
			d_keys_out,
			d_values_in,
			d_values_out,
			current_bit);

		// Process full tiles of tile_elements
		SizeT cta_offset = 0;
		while (cta_offset + TILE_ELEMENTS <= num_elements)
		{
			cta.ProcessTile<true>(cta_offset);
			cta_offset += TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-I/O
		if (cta_offset < num_elements)
		{
			SizeT remainder = num_elements - cta_offset;
			cta.ProcessTile<false>(cta_offset, remainder);
		}
	}
};







} // namespace cta
} // namespace radix_sort
} // namespace back40
BACK40_NS_POSTFIX
