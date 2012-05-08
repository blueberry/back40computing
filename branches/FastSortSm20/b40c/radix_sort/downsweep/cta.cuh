/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/******************************************************************************
 * CTA-processing functionality for radix sort downsweep scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/tex_vector.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>

#include <b40c/radix_sort/sort_utils.cuh>
#include <b40c/radix_sort/downsweep/kernel_policy.cuh>
#include <b40c/radix_sort/downsweep/tex_ref.cuh>
#include <b40c/radix_sort/spine/tex_ref.cuh>

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

	// Integer type for digit counters (to be packed into words of PackedCounters)
	typedef unsigned short DigitCounter;

	// Integer type for packing DigitCounters into columns of shared memory banks
	typedef typename util::If<
		(KernelPolicy::SMEM_8BYTE_BANKS),
		unsigned long long,
		unsigned int>::Type PackedCounter;

	enum {
		CURRENT_BIT 				= KernelPolicy::CURRENT_BIT,
		CURRENT_PASS 				= KernelPolicy::CURRENT_PASS,
		RADIX_BITS					= KernelPolicy::RADIX_BITS,
		RADIX_DIGITS 				= 1 << RADIX_BITS,
		KEYS_ONLY 					= util::Equals<ValueType, util::NullType>::VALUE,

		// Direction of flow though ping-pong buffers: (FLOP_TURN) ? (d_keys1 --> d_keys0) : (d_keys0 --> d_keys1)
		FLOP_TURN					= KernelPolicy::CURRENT_PASS & 0x1,

		// Whether or not to insert padding for exchanging keys
		BANK_PADDING 				= 1,

		LOG_THREADS 				= KernelPolicy::LOG_THREADS,
		THREADS						= 1 << LOG_THREADS,

		LOG_WARP_THREADS 			= CUB_LOG_WARP_THREADS(__CUB_CUDA_ARCH__),
		WARP_THREADS				= 1 << LOG_WARP_THREADS,

		LOG_WARPS					= LOG_THREADS - LOG_WARP_THREADS,
		WARPS						= 1 << LOG_WARPS,

		LOG_THREAD_ELEMENTS 		= KernelPolicy::LOG_THREAD_ELEMENTS,
		THREAD_ELEMENTS				= 1 << LOG_THREAD_ELEMENTS,

		LOG_TILE_ELEMENTS			= LOG_THREADS + LOG_THREAD_ELEMENTS,
		TILE_ELEMENTS				= 1 << LOG_TILE_ELEMENTS,

		BYTES_PER_COUNTER			= sizeof(DigitCounter),
		LOG_BYTES_PER_COUNTER		= util::Log2<BYTES_PER_COUNTER>::VALUE,

		PACKING_RATIO				= sizeof(PackedCounter) / sizeof(DigitCounter),
		LOG_PACKING_RATIO			= util::Log2<PACKING_RATIO>::VALUE,

		LOG_COUNTER_LANES			= CUB_MAX((RADIX_BITS - LOG_PACKING_RATIO), 0),				// Always at least one lane
		COUNTER_LANES				= 1 << LOG_COUNTER_LANES,

		// The number of packed counters per thread (plus one for padding)
		RAKING_SEGMENT				= COUNTER_LANES + 1,

		LOG_MEM_BANKS				= CUB_LOG_MEM_BANKS(__CUB_CUDA_ARCH__),
		MEM_BANKS					= 1 << LOG_MEM_BANKS,

		DIGITS_PER_SCATTER_PASS 	= THREADS / MEM_BANKS,
		SCATTER_PASSES 				= RADIX_DIGITS / DIGITS_PER_SCATTER_PASS,

		LOG_STORE_TXN_THREADS 		= LOG_MEM_BANKS,
		STORE_TXN_THREADS 			= 1 << LOG_STORE_TXN_THREADS,

		ELEMENTS_PER_TEX			= Textures<KeyType, ValueType, THREAD_ELEMENTS>::ELEMENTS_PER_TEX,

		THREAD_TEX_LOADS	 		= THREAD_ELEMENTS / ELEMENTS_PER_TEX,

		TILE_TEX_LOADS				= THREADS * THREAD_TEX_LOADS,
	};

	static const util::io::ld::CacheModifier 	LOAD_MODIFIER 		= KernelPolicy::LOAD_MODIFIER;
	static const util::io::st::CacheModifier 	STORE_MODIFIER 		= KernelPolicy::STORE_MODIFIER;
	static const ScatterStrategy 				SCATTER_STRATEGY 	= KernelPolicy::SCATTER_STRATEGY;

	// Key texture type
	typedef typename Textures<
		KeyType,
		ValueType,
		THREAD_ELEMENTS>::KeyTexType KeyTexType;

	// Value texture type
	typedef typename Textures<
		KeyType,
		ValueType,
		THREAD_ELEMENTS>::ValueTexType ValueTexType;


	/**
	 * Shared memory storage layout
	 */
	struct SmemStorage
	{
		SizeT						tex_offset;
		SizeT						tex_offset_limit;

		bool 						non_trivial_pass;
		util::CtaWorkLimits<SizeT> 	work_limits;

		SizeT 						base_digit_offset[RADIX_DIGITS];

		// Storage for scanning local ranks
		volatile PackedCounter		warpscan[WARPS][WARP_THREADS * 3 / 2];

		union {
			unsigned char			counter_base[1];
			DigitCounter			digit_counters[COUNTER_LANES + 1][THREADS][PACKING_RATIO];
			PackedCounter			raking_grid[THREADS][RAKING_SEGMENT];
			KeyType 				key_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
			ValueType 				value_exchange[TILE_ELEMENTS + (TILE_ELEMENTS >> LOG_MEM_BANKS)];
		};
	};


	/**
	 * Tile state
	 */
	struct Tile
	{
		KeyType 		keys[THREAD_ELEMENTS];					// Keys for this tile
		ValueType 		values[THREAD_ELEMENTS];				// Values for this tile
		DigitCounter	thread_prefixes[THREAD_ELEMENTS];		// For each key, the count of previous keys in this tile having the same digit
		int 			ranks[THREAD_ELEMENTS];					// For each key, the local rank within the tile
		unsigned int 	counter_offsets[THREAD_ELEMENTS];		// For each key, the byte-offset of its corresponding digit counter in smem
		SizeT			global_digit_base[THREAD_ELEMENTS];		// For each key, the global base scatter offset for the corresponding digit
	};


	//---------------------------------------------------------------------
	// Thread fields
	//---------------------------------------------------------------------

	// Shared storage for this CTA
	SmemStorage &smem_storage;

	KeyType *d_in_keys;
	KeyType	 *d_out_keys;

	ValueType *d_in_values;
	ValueType *d_out_values;

	PackedCounter *raking_segment;
	DigitCounter *bin_counter;

	// The global scatter base offset for each digit (valid in the
	// first RADIX_DIGITS threads)
	SizeT global_digit_base;

	int warp_id;
	volatile PackedCounter *warpscan;


	//---------------------------------------------------------------------
	// Helper structure for templated iteration
	//---------------------------------------------------------------------

	/**
	 * Iteration helper
	 */
	template <int COUNT, int MAX>
	struct Iterate
	{
		// DecodeKeys
		static __device__ __forceinline__ void DecodeKeys(Cta &cta,	Tile &tile)
		{
			// Compute byte offset of smem counter, starting with offset
			// of the thread's packed counter column
			unsigned int counter_offset = (threadIdx.x << (LOG_PACKING_RATIO + LOG_BYTES_PER_COUNTER));

			// Add in sub-counter offset
			counter_offset = Extract<
				KeyType,
				CURRENT_BIT + LOG_COUNTER_LANES,
				LOG_PACKING_RATIO,
				LOG_BYTES_PER_COUNTER>::SuperBFE(
					tile.keys[COUNT],
					counter_offset);

			// Add in row offset
			counter_offset = Extract<
				KeyType,
				CURRENT_BIT,
				LOG_COUNTER_LANES,
				LOG_THREADS + LOG_PACKING_RATIO + LOG_BYTES_PER_COUNTER>::SuperBFE(
					tile.keys[COUNT],
					counter_offset);

			DigitCounter* counter =
				(DigitCounter*) (cta.smem_storage.counter_base + counter_offset);

			// Load thread-exclusive prefix
			tile.thread_prefixes[COUNT] = *counter;

			// Store inclusive prefix
			*counter = tile.thread_prefixes[COUNT] + 1;

			// Remember counter offset
			tile.counter_offsets[COUNT] = counter_offset;

			// Next vector element
			Iterate<COUNT + 1, MAX>::DecodeKeys(cta, tile);
		}


		// ComputeLocalRanks
		static __device__ __forceinline__ void ComputeLocalRanks(Cta &cta, Tile &tile)
		{
			DigitCounter* counter =
				(DigitCounter*) (cta.smem_storage.counter_base + tile.counter_offsets[COUNT]);

			// Add in CTA exclusive prefix
			tile.ranks[COUNT] = tile.thread_prefixes[COUNT] + *counter;

			// Next vector element
			Iterate<COUNT + 1, MAX>::ComputeLocalRanks(cta, tile);
		}


		// ScatterRanked
		template <typename T>
		static __device__ __forceinline__ void ScatterRanked(
			Cta &cta,
			Tile &tile,
			T items[THREAD_ELEMENTS])
		{
			int offset = (BANK_PADDING) ?
				util::SHR_ADD(tile.ranks[COUNT], LOG_MEM_BANKS, tile.ranks[COUNT]) :
				tile.ranks[COUNT];

			((T*) cta.smem_storage.key_exchange)[offset] = items[COUNT];

			// Next vector element
			Iterate<COUNT + 1, MAX>::ScatterRanked(cta, tile, items);
		}

		// GatherShared
		template <typename T>
		static __device__ __forceinline__ void GatherShared(
			Cta &cta,
			Tile &tile,
			T items[THREAD_ELEMENTS])
		{
			int gather_offset =
				threadIdx.x +
				(BANK_PADDING ?
					(threadIdx.x >> LOG_MEM_BANKS) :
					0) +
				(BANK_PADDING ?
					(COUNT * THREADS) + ((COUNT * THREADS) >> LOG_MEM_BANKS) :
					(COUNT * THREADS));

			items[COUNT] = ((T*) cta.smem_storage.key_exchange)[gather_offset];

			// Next vector element
			Iterate<COUNT + 1, MAX>::GatherShared(cta, tile, items);
		}

		// DecodeBinOffsets
		static __device__ __forceinline__ void DecodeBinOffsets(Cta &cta, Tile &tile)
		{
			// Decode address of bin-offset in smem
			unsigned int byte_offset = Extract<
				KeyType,
				CURRENT_BIT,
				RADIX_BITS,
				util::Log2<sizeof(SizeT)>::VALUE>::SuperBFE(
					tile.keys[COUNT]);

			// Lookup global bin offset
			tile.global_digit_base[COUNT] = *(SizeT *)(((char *) cta.smem_storage.base_digit_offset) + byte_offset);

			// Next vector element
			Iterate<COUNT + 1, MAX>::DecodeBinOffsets(cta, tile);
		}

		// ScatterGlobal
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			Cta &cta,
			Tile &tile,
			T items[THREAD_ELEMENTS],
			T *d_out,
			const SizeT &guarded_elements)
		{
			int tile_element = threadIdx.x + (COUNT * THREADS);

			// Distribute if not out-of-bounds
			if ((guarded_elements >= TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				T* scatter = d_out + threadIdx.x + (THREADS * COUNT) + tile.global_digit_base[COUNT];
				util::io::ModifSTORtore<WRITE_MODIFIER>::St(items[COUNT], scatter);
			}

			// Next vector element
			Iterate<COUNT + 1, MAX>::ScatterGlobal(cta, tile, items, d_out, guarded_elements);
		}


		/**
		 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
		 * coalescing rules
		 */
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterPass(
			Cta &cta,
			T *exchange,
			T *d_out,
			const SizeT &valid_elements)
		{
			int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
			int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;

			int my_digit = (COUNT * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

			if (my_digit < RADIX_DIGITS) {

				int my_exclusive_scan = cta.smem_storage.warpscan[0][16 + my_digit - 1];
				int my_inclusive_scan = cta.smem_storage.warpscan[0][16 + my_digit];
				int my_digit_count = my_inclusive_scan - my_exclusive_scan;

				int my_carry = cta.smem_storage.base_digit_offset[my_digit] + my_exclusive_scan;
				int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

				while (my_aligned_offset < my_digit_count) {

					if ((my_aligned_offset >= 0) && (my_exclusive_scan + my_aligned_offset < valid_elements)) {

						T datum = exchange[my_exclusive_scan + my_aligned_offset];
						d_out[my_carry + my_aligned_offset] = datum;
					}
					my_aligned_offset += STORE_TXN_THREADS;
				}
			}

			// Next scatter pass
			Iterate<COUNT + 1, MAX>::AlignedScatterPass(cta, exchange, d_out, valid_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int MAX>
	struct Iterate<MAX, MAX>
	{
		// DecodeKeys
		static __device__ __forceinline__ void DecodeKeys(Cta &cta, Tile &tile) {}

		// ComputeLocalRanks
		static __device__ __forceinline__ void ComputeLocalRanks(Cta &cta, Tile &tile) {}

		// ScatterRanked
		template <typename T>
		static __device__ __forceinline__ void ScatterRanked(Cta &cta, Tile &tile, T items[THREAD_ELEMENTS]) {}

		// GatherShared
		template <typename T>
		static __device__ __forceinline__ void GatherShared(Cta &cta, Tile &tile, T items[THREAD_ELEMENTS]) {}

		// DecodeBinOffsets
		static __device__ __forceinline__ void DecodeBinOffsets(Cta &cta, Tile &tile) {}

		// ScatterGlobal
		template <typename T>
		static __device__ __forceinline__ void ScatterGlobal(Cta &cta, Tile &tile, T items[THREAD_ELEMENTS], T *d_out, const SizeT &guarded_elements) {}

		// AlignedScatterPass
		template <typename T>
		static __device__ __forceinline__ void AlignedScatterPass(Cta &cta, T *exchange, T *d_out, const SizeT &valid_elements) {}
	};




	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage 	&smem_storage,
		KeyType 		*d_keys0,
		KeyType 		*d_keys1,
		ValueType 		*d_values0,
		ValueType 		*d_values1,
		SizeT 			*d_spine) :
			smem_storage(smem_storage),
			d_in_keys(FLOP_TURN ? d_keys1 : d_keys0),
			d_out_keys(FLOP_TURN ? d_keys0 : d_keys1),
			d_in_values(FLOP_TURN ? d_values1 : d_values0),
			d_out_values(FLOP_TURN ? d_values0 : d_values1),
			raking_segment(smem_storage.raking_grid[threadIdx.x])
	{
		int counter_lane = threadIdx.x & (COUNTER_LANES - 1);
		int sub_counter = threadIdx.x >> (LOG_COUNTER_LANES);
		bin_counter = &smem_storage.digit_counters[counter_lane][0][sub_counter];

		// Initialize warpscan identity regions
		warp_id = threadIdx.x >> 5;
		warpscan = &smem_storage.warpscan[warp_id][16 + (threadIdx.x & 31)];
		warpscan[-16] = 0;

		if ((THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS)) {

			// Read base_digit_offset in parallel
			int spine_bin_offset = (gridDim.x * threadIdx.x) + blockIdx.x;
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

			#pragma unroll
			for (int PACK = 0; PACK < THREAD_TEX_LOADS; PACK++) {

				vectors[PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						TexKeys<KeyTexType>::ref1 :
						TexKeys<KeyTexType>::ref0,
					tex_offset + (threadIdx.x * THREAD_TEX_LOADS) + PACK);
			}
		} else {

			// Guarded loads with default assignment of -1 to out-of-bound keys
			util::io::LoadTile<
				0,									// log loads per tile
				LOG_THREAD_ELEMENTS,
				THREADS,
				READ_MODIFIER,
				false>::LoadValid(
					(KeyType (*)[THREAD_ELEMENTS]) tile.keys,
					d_in_keys,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements,
					KeyType(-1));
		}
	}

	/**
	 * Load tile of values
	 */
	__device__ __forceinline__ void LoadValues(
		SizeT tex_offset,
		const SizeT &guarded_elements,
		Tile &tile)
	{
		if (guarded_elements >= TILE_ELEMENTS) {

			// Unguarded loads through tex
			ValueTexType *vectors = (ValueTexType*) tile.values;

			#pragma unroll
			for (int PACK = 0; PACK < THREAD_TEX_LOADS; PACK++) {

				vectors[PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						TexValues<ValueTexType>::ref1 :
						TexValues<ValueTexType>::ref0,
					tex_offset + (threadIdx.x * THREAD_TEX_LOADS) + PACK);
			}

		} else {
			// Guarded loads with default assignment of -1 to out-of-bound values
			util::io::LoadTile<
				0,									// log loads per tile
				LOG_THREAD_ELEMENTS,
				THREADS,
				READ_MODIFIER,
				false>::LoadValid(
					(ValueType (*)[THREAD_ELEMENTS]) tile.values,
					d_in_values,
					(tex_offset * ELEMENTS_PER_TEX),
					guarded_elements);
		}
	}


	/**
	 * Scan shared memory counters
	 LO
	__device__ __forceinline__ void ScanCounters(Tile &tile)
	{
		// Upsweep reduce
		PackedCounter raking_partial = util::reduction::SerialReduce<RAKING_SEGMENT>::Invoke(
			raking_segment);

		// Warpscan
		PackedCounter partial = raking_partial;
		warpscan[0] = partial;

		warpscan[0] = partial =
			partial + warpscan[0 - 1];
		warpscan[0] = partial =
			partial + warpscan[0 - 2];
		warpscan[0] = partial =
			partial + warpscan[0 - 4];
		warpscan[0] = partial =
			partial + warpscan[0 - 8];
		warpscan[0] = partial =
			partial + warpscan[0 - 16];

		// Barrier
		__syncthreads();

		// Scan across warpscan totals
		PackedCounter warpscan_totals = 0;

		#pragma unroll
		for (int WARP = 0; WARP < WARPS; WARP++) {

			// Add totals from all previous warpscans into our partial
			PackedCounter warpscan_total = smem_storage.warpscan[WARP][(WARP_THREADS * 3 / 2) - 1];
			if (warp_id == WARP) {
				partial += warpscan_totalLO
			}

			// Increment warpscan totals
			warpscan_totals += warpscan_total;
		}

		// Add lower totals from all warpscans into partial's upper
		#pragma unroll
		for (int PACKED = 1; PACKED < PACKING_RATIO; PACKED++) {
			partial += warpscan_totals << (16 * PACKED);
		}

		// Downsweep scan with exclusive partial
		PackedCounter exclusive_partial = partial - raking_partial;
		util::scan::SerialScan<RAKING_SEGMENT>::Invoke(
			raking_segment,
			exclusive_partial);
	}


	/**
	 * Truck along associated values.  (Specialized for key-value passes.)
	 */
	template <bool IS_KEYS_ONLY, int DUMMY = 0>
	struct TruckValues
	{
		static __device__ __forceinline__ void Invoke(
			SizeT tex_offset,
			const SizeT &guarded_elements,
			Cta &cta,
			Tile &tile)
		{
			// Load tile of values
			cta.LoadValues(tex_offset, guarded_elements, tile);

			__syncthreads();

			// Scatter values shared
			Iterate<0, THREAD_ELEMENTS>::ScatterRanked(cta, tile, tile.values);

			__syncthreads();

			// Gather values from shared memory and scatter to global
			if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

				// Use explicitly warp-aligned scattering of values from smem
				Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
					cta,
					cta.smem_storage.value_exchange,
					cta.d_out_values,
					guarded_elements);

			} else {

				// Gather values shared
				Iterate<0, THREAD_ELEMENTS>::GatherShared(cta, tile, tile.values);

				// Scatter to global
				Iterate<0, THREAD_ELEMENTS>::ScatterGlobal(
					cta,
					tile,
					tile.values,
					cta.d_out_values,
					guarded_elements);
			}
		}
	};


	/**
	 * Truck along associated values.  (Specialized for keys-only passes.)
	 */
	template <int DUMMY>
	struct TruckValues<true, DUMMY>
	{
		static __device__ __forceinline__ void Invoke(
			SizeT tex_offset,
			const SizeT &guarded_elements,
			Cta &cta,
			Tile &tile)
		{
			// do nothing
		}
	};


	/**
	 * Gather keys from smem and scatter to global
	 */
	__device__ __forceinline__ void GatherScatterKeys(
		Tile &tile,
		const SizeT &guarded_elements)
	{
		if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

			// Use explicitly warp-aligned scattering of keys from smem
			Iterate<0, SCATTER_PASSES>::AlignedScatterPass(
				*this,
				smem_storage.key_exchange,
				d_out_keys,
				guarded_elements);

		} else {

			// Gather keys
			Iterate<0, THREAD_ELEMENTS>::GatherShared(*this, tile, tile.keys);

			// Decode global scatter offsets
			Iterate<0, THREAD_ELEMENTS>::DecodeBinOffsets(*this, tile);

			// Scatter to global
			Iterate<0, THREAD_ELEMENTS>::ScatterGlobal(
				*this,
				tile,
				tile.keys,
				d_out_keys,
				guarded_elements);
		}
	}


	/**
	 * Reset shared memory digit counters
	 */
	__device__ __forceinline__ void ResetCounters()
	{
		#pragma unroll
		for (int LANE = 0; LANE < COUNTER_LANES + 1; LANE++) {
			*((PackedCounter*) smem_storage.digit_counters[LANE][threadIdx.x]) = 0;
		}
	}


	/**
	 * Update global scatter offsets for each digit
	 */
	__device__ __forceinline__ void UpdateDigitScatterOffsets()
	{
		if ((THREADS == RADIX_DIGITS) || (threadIdx.x < RADIX_DIGITS)) {

			DigitCounter bin_inclusive = bin_counter[THREADS * PACKING_RATIO];
			smem_storage.warpscan[0][16 + threadIdx.x] = bin_inclusive;
			PackedCounter bin_exclusive = smem_storage.warpscan[0][16 + threadIdx.x - 1];

			global_digit_base -= bin_exclusive;
			smem_storage.base_digit_offset[threadIdx.x] = global_digit_base;
			global_digit_base += bin_inclusive;
		}
	}


	/**
	 * Process tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT tex_offset,
		const SizeT &guarded_elements = TILE_ELEMENTS)
	{
		// State for the current tile
		Tile tile;

		// Load tile of keys
		LoadKeys(tex_offset, guarded_elements, tile);

		__syncthreads();

		// Reset shared memory digit counters
		ResetCounters();

		// Decode bins and update counters
		Iterate<0, THREAD_ELEMENTS>::DecodeKeys(*this, tile);

		__syncthreads();

		// Scan shared memory counters
		ScanCounters(tile);

		__syncthreads();

		// Update global scatter offsets for each digit
		UpdateDigitScatterOffsets();

		// Extract the local ranks of each key
		Iterate<0, THREAD_ELEMENTS>::ComputeLocalRanks(*this, tile);

		__syncthreads();

		// Scatter keys to shared memory in sorted order
		Iterate<0, THREAD_ELEMENTS>::ScatterRanked(*this, tile, tile.keys);

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
		