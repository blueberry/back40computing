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
 * CTA-processing functionality for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/tex_vector.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/ns_umbrella.cuh>

#include <b40c/radix_sort/sort_utils.cuh>
#include <b40c/radix_sort/cta_radix_rank.cuh>
#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/tex_ref.cuh>

B40C_NS_PREFIX
namespace b40c {
namespace radix_sort {
namespace downsweep {


/**
 * Partitioning downsweep scan CTA
 */
template <
	typename KernelPolicy,
	typename SizeT,
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
	static const ScatterStrategy 				SCATTER_STRATEGY 	= KernelPolicy::SCATTER_STRATEGY;

	enum {
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,
		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,

		CURRENT_BIT 				= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 				= KernelPolicy::CURRENT_PASS,

		// Direction of flow though ping-pong buffers: (FLOP_TURN) ? (d_keys1 --> d_keys0) : (d_keys0 --> d_keys1)
		FLOP_TURN					= KernelPolicy::CURRENT_PASS & 0x1,

		LOG_CTA_THREADS 			= KernelPolicy::LOG_CTA_THREADS,
		CTA_THREADS					= 1 << LOG_CTA_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_CTA_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_THREAD_ELEMENTS 		= KernelPolicy::LOG_THREAD_ELEMENTS,
		KEYS_PER_THREAD				= 1 << LOG_THREAD_ELEMENTS,

		LOG_TILE_ELEMENTS			= LOG_CTA_THREADS + LOG_THREAD_ELEMENTS,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		BYTES_PER_SIZET				= sizeof(SizeT),
		LOG_BYTES_PER_SIZET			= util::Log2<BYTES_PER_SIZET>::VALUE,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		// Whether or not to insert padding for exchanging keys. (Padding is
		// worse than bank conflicts on GPUs that need two-phase scattering)
		PADDED_EXCHANGE 			= (SCATTER_STRATEGY != SCATTER_WARP_TWO_PHASE),
		PADDING_ELEMENTS			= (PADDED_EXCHANGE) ? (TILE_ELEMENTS >> LOG_MEM_BANKS) : 0,

		DIGITS_PER_SCATTER_PASS 	= CTA_THREADS / MEM_BANKS,
		SCATTER_PASSES 				= RADIX_DIGITS / DIGITS_PER_SCATTER_PASS,

		LOG_STORE_TXN_THREADS 		= LOG_MEM_BANKS,
		STORE_TXN_THREADS 			= 1 << LOG_STORE_TXN_THREADS,

		ELEMENTS_PER_TEX			= Textures<KeyType, ValueType, KEYS_PER_THREAD>::ELEMENTS_PER_TEX,

		THREAD_TEX_LOADS	 		= KEYS_PER_THREAD / ELEMENTS_PER_TEX,

		TILE_TEX_LOADS				= CTA_THREADS * THREAD_TEX_LOADS,
	};

	// Texture types
	typedef Textures<KeyType, ValueType, KEYS_PER_THREAD> 	Textures;
	typedef typename Textures::KeyTexType 					KeyTexType;
	typedef typename Textures::ValueTexType 				ValueTexType;

	// CtaRadixRank utility type
	typedef CtaRadixRank<
		LOG_CTA_THREADS,
		RADIX_BITS,
		CURRENT_BIT,
		KernelPolicy::SMEM_CONFIG> CtaRadixRank;

	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		SizeT							tex_offset;
		SizeT							tex_offset_limit;

		util::CtaWorkLimits<SizeT> 		work_limits;

		union
		{
			unsigned char				digit_offset_bytes[1];
			SizeT 						digit_offsets[RADIX_DIGITS];
		};

		union
		{
			typename CtaRadixRank::SmemStorage	ranking_storage;
			UnsignedBits						key_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
			ValueType 							value_exchange[TILE_ELEMENTS + PADDING_ELEMENTS];
		};
	};


	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage 				&smem_storage;

	// Input and output device pointers
	UnsignedBits 				*d_in_keys;
	UnsignedBits				*d_out_keys;
	ValueType 					*d_in_values;
	ValueType 					*d_out_values;

	// The global scatter base offset for each digit (valid in the first RADIX_DIGITS threads)
	SizeT 						my_digit_offset;


	//---------------------------------------------------------------------
	// Helper structure for templated iteration.  (NVCC currently won't
	// unroll loops with "unexpected control flow".)zeT>::ref, spine_bin_offset);
		}
	}


	/**
	 * Load tile of keys
	 */
	__device__ _Iterate
	 */
	template <int COUNT, int MAX>
	struct Iterate
	{
		/**
		 * Scatter items to global memory
		 */
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			T 				items[KEYS_PER_THREAD],
			SizeT			digit_offsets[KEYS_PER_THREAD],
			T 				*d_out,
			const SizeT 	&guarded_elements)
		{
			// Scatter if not out-of-bounds
			int tile_element = threadIdx.x + (COUNT * CTA_THREADS);
			T* scatter = d_out + threadIdx.x + (COUNT * CTA_THREADS) + digit_offsets[COUNT];
of-bounds
			if ((guarded_elements >= TILE_ELEMENTS) || (tile_element < guarded_e
			{ents)NT];
				util::io::ModifSTORtore<WRITE_MODIFIER>::St(items[COUNT], scatter);
			Iterate next element
			Iterate<COUNT + 1, MAX>::ScatterGlobal(items, digit_offsets, d_out, guarded_elements);
		}


		/**
		 * Scatter items to global memory
		 */
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			T 				items[KEYS_PER_THREAD],
			unsigned int 	ranks[KEYS_PER_THREAD],
			SizeT			digit_offsets[KEYS_PER_THREAD],
			T 				*d_out,
			const SizeT 	&guarded_elements)
		{
			// Scatter if not out-of-bounds
			T* scatter = d_out + ranks[COUNT] + digit_offsets[COUNT];

			if ((guarded_elements >= TILE_ELEMENTS) || (ranks[COUNT] < guarded_elements))
			{e[COUNT];
				util::io::ModifSTORtore<WRITE_MODIFIER>::St(items[COUNT], scatter);
			Iterate next element
			Iterate<COUNT + 1, MAX>::ScatterGlobal(items, ranks, digit_offsetile, items, d_out, guarded_elements);
		}


		/**
		 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
		 * coalescing rules
		 */
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterSmemStorage 	&smem_storage,
			T 				*buffer,
			T 				*d_out,
			const SizeT 	&valid_elements)
		{
			typedef typename CtaRadixRank::PackedCounter PackedCounter;

			int store_txn_idx 		= threadIdx.x & (STORE_TXN_THREADS - 1);
			int store_txn_digit 	= threadIdx.x >> LOG_STORE_TXN_THREADS;
			int my_digit 			my_digit = (COUNT * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

			if (my_digit < RADI
			{IGITS) {

				int my_exclus	= smem_storage.ranking_storage.warpscan[0][(WARP_THREADS / 2) + my_digit - 1];
				int my_inclusive_scan 	= smem_storage.ranking_storage.warpscan[0][(WARP_THREADS / 2) + my_digit];
				int my_digit_count 		= my_inclusive_scan - my_exclusive_scan;
				int my_carry 			= smem_storage.digit_offsets[my_digit] + my_exclusive_scan;
				int my_aligned_offset 	d_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

				while (my_aligned_offset < my_dig
				{
					if ((my_aligned_offset >= 0) && (my_exclusive_scan + my_aligned_offset < valid_elements))
					{
						int gather_offset = my_exclusive_scan + my_aligned_offset;
						if (PADDED_EXCHANGE) gather_offset = util::SHR_ADD(gather_offset, LOG_MEM_BANKS, gather_offset);ments)) {

						Tbuffer exchange[my_exclusive_scan + my_aligned_offset];
						d_out[my_carry + my_aligned_offset] = datum;
					}
					my_aligned_offset += STORE_TXN_THREADS;
				}
			}

			// Next scatter pass
			Iterate<COUNT + 1, MAX>::AlignedScasmem_storage, buffer, d_out, valid_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// ScatterGlobal
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(T[KEYS_PER_THREAD], SizeT[KEYS_PER_THREAD], T*, const SizeT &) {}

		// ScatterGlobal
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(T[KEYS_PER_THREAD], unsigned int[KEYS_PER_THREAD], SizeT[KEYS_PER_THREAD], T*, const SizeT &) {}

		// AlignedScatterPassles
		 */
		template <typename T>
		static __device__ __forceinline__ void AlignedScaSmemStorage&, T*, T*, const SizeT&) {}
	};Dim.x * threadIdx.x) + blockIdx.x;
			global_digit_base = tex1Dfetch(spine::TexSpine<SizeT>::ref, spine_bin_offset);
		}
	}


	/**
	 * Load tile of keys
	 */
	__device__ __forceinline__ void LoadKeys(
		SizeT tex_offset,
		const SizeT &guarded_elements,
		Tile &tile)
	{
		if (guarded_elements >= TILE_ELEMENTS) {

			// Unguarded loads through tex
			KeyTexType *vectors = (KeyTexType *) tile.keys;

			#pragma unreinterpret_cast<UnsignedBits*>(FLOP_TURN ? d_keys1 : d_keys0)),
			d_out_keys(reinterpret_cast<UnsignedBits*>(FLOP_TURN ? d_keys0 : d_keys1)PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						TexKeys<KeyTexType>::ref1 :
						TexKeys<KeyTexType>:
	{
		if ((CTA_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
		{
			// Read digit scatter base (in parallel)
			int spine_digit_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
			my_digit_offset = d_spine[spine_digit_offset];
		}
	}


	/**
	 * Perform a bit-wise twiddling transformation on keys
	 */
	template <UnsignedBits TwiddleOp(UnsignedBits)>
	__device__ __forceinline__ void TwiddleKeys(
		UnsignedBits converted_keys[KEYS_PER_THREAD])
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{
			converted_keys[KEY] = TwiddleOp(converted_keys[KEY]);
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
			int offset = (PADDED_EXCHANGE) ?
				util::SHR_ADD(ranks[KEY], LOG_MEM_BANKS, ranks[KEY]) :
				ranks[KEY];

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
		UnsignedBits 	converted_keys[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD])
	{
		#pragma unroll
		for (int KEY = 0; KEY < KEYS_PER_THREAD; KEY++)
		{tile)
		{
			// Decode address of bin-offset in smem
			unsigned int byte_offset = ExtrCURRENT_BIT,
				RADIX_BITS,
				LOG_BYTES_PER_SIZET>(converted_keys[KEY]);

			// Lookup base digit offset from shared memory
			digit_offsets[KEY] = *(SizeT *)(smem_storage.digit_offset_bytes + byte_offset);
		}
	}

	/**
	 * Load tile of keys from global memory
	 */
	__device__ __forceinline__ void LoadKeys(
		UnsignedBits 	converted_keys[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		const SizeT 	&guarded_elements)
	{
		if ((LOAD_MODIFIER == util::io::ld::tex) && (guarded_elements >= TILE_ELEMENTS))
		{
			// Unguarded loads through tex
			KeyTexType *vectors = (KeyTexType *) converted_et + (threadIdx.x * THREAD_TEX_LOADS) + PACK);
			}

		} else {
			// Guarded l
			{s with default assignment of -1 to out-of-bound values
	 TexKeys<KeyTexType>::ref1 : og loads per tile
				LOG_THREAD_ELEMENTS,
				THREADS,
				READ_MODIFIER,
				false>::LoadVali
		else
		{(ValueType (*)[THREAD_ELEMENTS]) tile.values,
		MAX_KEY	d_in_values,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements);
		}
	}


	/**
	 * Scan shared memorCTA_THREADS,
				LOAD_MODIFIER,
				false>::LoadValid(
					(UnsignedBits (*)[KEYS_PER_THREAD]) converted_keys,
					d_in_keys,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements,
					MAX_KEY);
		}
	}


	/**
	 * Load tile of values from global memory
	 */
	__device__ __forceinline__ void LoadValues(
		ValueType 		values[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		const SizeT 	&guarded_elements)
	{
		if ((LOAD_MODIFIER == util::io::ld::tex) &&
			(util::NumericTraits<ValueType>::BUILT_IN) &&
			(guarded_elements >= TILE_ELEMENTS))
		{- 2];
		warpscan[0] = partial =
			partial + warpscan[0 - 4];
		warpscan[0] = value+ (threadIdx.x * THREAD_TEX_LOADS) + PACK);
			}

		} else {
			// Guarded l
			{s with default assignment of -1 to out-of-bound values
	 TexValues<ValueTexType>::ref1 : TexValues<Valuer tile
				LOG_THREAD_ELEMENTS,
				THREADS,
				READ_MODIFIER,
				false>::LoadVali
		else
		{(ValueType (*)[THREAD_ELEMENTS]) tile.values,
					d_in_values,
			values
			util::io::LoadTile<
				0,									// log loads per tile
				LOG_THREAD_ELEMENTS,
				CTA_THREADS,
				LOAD_MODIFIER,
				false>::LoadValid(
					(ValueType (*)[KEYS_PER_THREAD]) values,
					d_in_values,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements);
		}
	}


	/**
	 * Gather keys from smem, decode base digit offsets for keys,4
	 * and scatter to global
	 */
	__device__ __forceinline__ void ScatterKeys(
		UnsignedBits 	converted_keys[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],		// (out parameter)
		unsigned int 	ranks[KEYS_PER_THREAD],
		const SizeT 	&guarded_elements)
	{
		if (SCATTER_STRATEGY == SCATTER_DIRECT)
		{
			// Compute scatter offsets
			DecodeDigitOffsets(converted_keys, digit_offsets);

			// Twiddle keys before outputting
			TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(converted_keys);

			// Scatter keys directly to global memory
			Iterate<0, KEYS_PER_THREAD>::ScatterGlobal(
				converted_keys,
				ranks,
				digit_offsets,
				d_out_keys,
				guarded_elements);
		}
		else
		{
			__syncthreads();

			// Exchange keys through shared memory for better write-coalescing
			ScatterRanked(ranks, converted_keys, smem_storage.key_exchange);

			__syncthreads();

			if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE)
			{
				// Twiddle keys before outputting
				TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(converted_keys);

				// Use explicitly warp-aligned scattering of keys from shared memory
				Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
					smem_storage,
					smem_storage.key_exchange,
					d_out_keys,
					guarded_elements);
			}
			else
			{
				// Gather keys from shared memory
				GatherShared(converted_keys, smem_storage.key_exchange);

				// Compute scatter offsets
				DecodeDigitOffsets(converted_keys, digit_offsets);

				// Twiddle keys before outputting
				TwiddleKeys<KeyTraits<KeyType>::TwiddleOut>(converted_keys);

				// Scatter keys to global memory
				Iterate<0, KEYS_PER_THREAD>::ScatterGlobal(
					converted_keys,
					digit_offsets,
					d_out_keys,
					guarded_elements);
			}
		}
	}


	/**
	 * Truck along associated values
	 */
	template <typename _ValueType>
	__device__ __forceinline__ void GatherScatterValues(
		_ValueType 		values[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		const SizeT 	&guarded_elements)
	{
		// Load tile of values
		LoadValues(values, tex_offset, guarded_elements);

		if (SCATTER_STRATEGY == SCATTER_DIRECT)
		{
			// Scatter values directly to global memory
			Iterate<0, KEYS_PER_THREAD>::ScatterGlobal(
				values,
				ranks,
				digit_offsets,
				d_out_values,
				guarded_elements);
		}
		else
		{
			__syncthreads();

			// Exchange values through shared memory for better write-coalescing
			ScatterRanked(ranks, values, smem_storage.value_exchange);

			__syncthreads();
Shared
		template <typename T>
		static __device__ __forceinline__ void GatherShared(Cta &cta, Tile &tile, T items[THREhared memory
				Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
					smem_storage,
					smem_storage.value_exchange,
					d_out_values,
					guarded_elements);
			}
			else
			{
				// Gather values from shared
				GatherShared(values, smem_storage.value_exchange);

				// Scatter to global memory
				Iterate<0, KEYS_PER_THREAD>::ScatterGlobal(
					values,
					digit_offsets,
					d_out_values,
					guarded_elements);
			}
		}
	}


	/**
	 * Truck along associated values (specialized for key-only sorting)
	 */
	__device__ __forceinline__ void GatherScatterValues(
		util::NullType	values[KEYS_PER_THREAD],
		SizeT 			digit_offsets[KEYS_PER_THREAD],
		unsigned int 	ranks[KEYS_PER_THREAD],
		SizeT 			tex_offset,
		const SizeT 	&guarded_elements)
	{n[0][16 + threadIdx.x] = bin_inclusive;
			PackedCounter bin_exclusive = smem_storage.warpscan[0][16 + threadIdx.x - 1];

			global_digit_base -= bin_exclTile data
		UnsignedBits 	converted_keys[KEYS_PER_THREAD];		// Keys
		ValueType 		values[KEYS_PER_THREAD];				// Values
		unsigned int	ranks[KEYS_PER_THREAD];					// For each key, the local rank within the CTA
		SizeT 			digit_offsets[KEYS_PER_THREAD];			// For each key, the global scatter base offset of the corresponding digit

		// Load tile of keys and twiddle bits if necessary
		LoadKeys(converted_keys, tex_offset, guarded_elements);

		__syncthreads();

		// Twiddle keys
		TwiddleKeys<KeyTraits<KeyType>::TwiddleIn>(converted_keys);

		// Rank keys
		unsigned int inclusive_digit_count;						// Inclusive digit count for each digit (corresponding to thread-id)
		unsigned int exclusive_digit_count;						// Exclusive digit count for each digit (corresponding to thread-id)

		CtaRadixRank::RankKeys(
			smem_storage.ranking_storage,
			converted_keys,
			ranks,
			inclusive_digit_count,
			exclusive_digit_count);

		// Update global scatter base offsets for each digit
		if ((CTA_THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS))
		{
			my_digit_offset -= exclusive_digit_count;
			smem_storage.digit_offsets[threadIdx.x] = my_digit_offset;
			my_digit_offset += inclusive_digit_count;
		}

		__syncthreads();

		// Scatter keys
		ScatterKeys(converted_keys, digit_offsets, ranks, guarded_elements);

		// Gather/scatter values
		GatherScatterValues(values, digit_offsets, ranks, tex_offset, guarded_elementsked(*this, tile, tile.keys);

		__syncthreads();

		// Gather keys from shared memory and scatter to global
		GatherScatterKeys(tile, guarded_elements);

		// Truck along values (if applicable)
		TruckValues<KEYS_ONLY>::Invoke(tex_offset, guarded_elements, *this, tile);
	}


	/**
	 * Process work range of tiles
	 */
	__device__ __forceinline__ void ProcessWorkRange(
		util::CtaWorkLimits<SizeT> &work_limits)
	{
		// Make sure we get a local copy of the cta's offset (work_limits may be in smem)
		SizeT tex_offset = smem_storage.tex_offset;

		// Process full tiles of tile_elements
		while (tex_offset < smem_storage.tex_offset_limit) {
			ProcessTile(tex_offset);
		