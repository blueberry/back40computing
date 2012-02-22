/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * Abstract tile-processing functionality for partitioning downsweep scan
 * kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>
#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/scan/soa/cooperative_soa_scan.cuh>

namespace b40c {
namespace partition {
namespace downsweep {



/**
 * Templated texture reference for keys
 */
template <typename KeyVectorType>
struct KeysTex
{
	static texture<KeyVectorType, cudaTextureType1D, cudaReadModeElementType> ref0;
	static texture<KeyVectorType, cudaTextureType1D, cudaReadModeElementType> ref1;
};
template <typename KeyVectorType>
texture<KeyVectorType, cudaTextureType1D, cudaReadModeElementType> KeysTex<KeyVectorType>::ref0;
template <typename KeyVectorType>
texture<KeyVectorType, cudaTextureType1D, cudaReadModeElementType> KeysTex<KeyVectorType>::ref1;

/**
 * Templated texture reference for values
 */
template <typename ValueVectorType>
struct ValuesTex
{
	static texture<ValueVectorType, cudaTextureType1D, cudaReadModeElementType> ref0;
	static texture<ValueVectorType, cudaTextureType1D, cudaReadModeElementType> ref1;
};
template <typename ValueVectorType>
texture<ValueVectorType, cudaTextureType1D, cudaReadModeElementType> ValuesTex<ValueVectorType>::ref0;
template <typename ValueVectorType>
texture<ValueVectorType, cudaTextureType1D, cudaReadModeElementType> ValuesTex<ValueVectorType>::ref1;


/**
 * Tile
 *
 * Abstract class
 */
template <
	typename KernelPolicy,
	typename DerivedTile>
struct Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::KeyType 					KeyType;
	typedef typename KernelPolicy::ValueType 				ValueType;
	typedef typename KernelPolicy::SizeT 					SizeT;

	typedef typename util::VecType<
		KeyType,
		KernelPolicy::PACK_SIZE>::Type 						KeyVectorType;
	typedef typename util::VecType<
		ValueType,
		KernelPolicy::PACK_SIZE>::Type 						ValueVectorType;

	typedef DerivedTile Dispatch;

	enum {
		LOAD_VEC_SIZE 				= KernelPolicy::LOAD_VEC_SIZE,

		LOG_PACKS_PER_LOAD			= KernelPolicy::LOG_LOAD_VEC_SIZE - KernelPolicy::LOG_PACK_SIZE,
		PACKS_PER_LOAD				= 1 << LOG_PACKS_PER_LOAD,

		LOG_RAKING_THREADS 			= KernelPolicy::LOG_RAKING_THREADS,
		RAKING_THREADS 				= 1 << LOG_RAKING_THREADS,

		WARP_THREADS				= B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH),

		LOG_MEM_BANKS				= B40C_LOG_MEM_BANKS(KernelPolicy::CUDA_ARCH)
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// The keys (and values) this thread will read this tile
	KeyType 			keys[LOAD_VEC_SIZE];
	ValueType 			values[LOAD_VEC_SIZE];
	unsigned short		prefixes[LOAD_VEC_SIZE];
	int					counter_offsets[LOAD_VEC_SIZE];
	SizeT				bin_offsets[LOAD_VEC_SIZE];


	//---------------------------------------------------------------------
	// IterateTileElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int VEC, int DUMMY = 0>
	struct IterateTileElements
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta,	Tile *tile)
		{
			int sub_counter = util::BFE(
				tile->keys[VEC],
				KernelPolicy::CURRENT_BIT + KernelPolicy::LOG_SCAN_BINS - 1,
				1);

			int lane = util::BFE(
				tile->keys[VEC],
				KernelPolicy::CURRENT_BIT,
				KernelPolicy::LOG_SCAN_BINS - 1);

			tile->counter_offsets[VEC] = (lane * KernelPolicy::THREADS * 2) + sub_counter;

			// Load thread-exclusive prefix
			tile->prefixes[VEC] = cta->counters[tile->counter_offsets[VEC]];

			// Store inclusive prefix
			cta->counters[tile->counter_offsets[VEC]] = tile->prefixes[VEC] + 1;

			// Next vector element
			IterateTileElements<VEC + 1>::DecodeKeys(cta, tile);
		}


		// ComputeRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// Add in CTA exclusive prefix
			tile->prefixes[VEC] += cta->counters[tile->counter_offsets[VEC]];

			// Next vector element
			IterateTileElements<VEC + 1>::ComputeRanks(cta, tile);
		}


		// ScatterRanked
		template <bool BANK_PADDING, typename Cta, typename Tile, typename T>
		static __device__ __forceinline__ void ScatterRanked(
			Cta *cta,
			Tile *tile,
			T items[LOAD_VEC_SIZE])
		{
			int offset = (BANK_PADDING) ?
				util::SHR_ADD(tile->prefixes[VEC], LOG_MEM_BANKS, tile->prefixes[VEC]) :
				tile->prefixes[VEC];

			((T*) cta->smem_storage.key_exchange)[offset] = items[VEC];

			// Next vector element
			IterateTileElements<VEC + 1>::template ScatterRanked<BANK_PADDING>(cta, tile, items);
		}

		// GatherShared
		template <bool BANK_PADDING, typename Cta, typename Tile, typename T>
		static __device__ __forceinline__ void GatherShared(
			Cta *cta,
			Tile *tile,
			T items[LOAD_VEC_SIZE])
		{
			const int LOAD_OFFSET = (BANK_PADDING) ?
				(VEC * KernelPolicy::THREADS) + ((VEC * KernelPolicy::THREADS) >> LOG_MEM_BANKS) :
				(VEC * KernelPolicy::THREADS);

			KeyType *base_gather_offset = (BANK_PADDING) ?
				((T*) cta->smem_storage.key_exchange) + threadIdx.x + (threadIdx.x >> LOG_MEM_BANKS) :
				((T*) cta->smem_storage.key_exchange) + threadIdx.x;

			items[VEC] = base_gather_offset[LOAD_OFFSET];

			// Next vector element
			IterateTileElements<VEC + 1>::template GatherShared<BANK_PADDING>(cta, tile, items);
		}

		// DecodeBinOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeBinOffsets(Cta *cta, Tile *tile)
		{
			// Decode bin from key
			int bin = util::BFE(tile->keys[VEC], KernelPolicy::CURRENT_BIT, KernelPolicy::LOG_BINS);

			// Lookup global bin offset
			tile->bin_offsets[VEC] = cta->smem_storage.bin_carry[bin];

			// Next vector element
			IterateTileElements<VEC + 1>::DecodeBinOffsets(cta, tile);
		}

		// ScatterGlobal
		template <typename Cta, typename Tile, typename T>
		static __device__ __forceinline__ void ScatterGlobal(
			Cta *cta,
			Tile *tile,
			T items[LOAD_VEC_SIZE],
			T *d_out,
			const SizeT &guarded_elements)
		{
			int tile_element = threadIdx.x + (VEC * KernelPolicy::THREADS);

			// Distribute if not out-of-bounds
			if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					items[VEC],
					d_out + threadIdx.x + (KernelPolicy::THREADS * VEC) + tile->bin_offsets[VEC]);
			}

			// Next vector element
			IterateTileElements<VEC + 1>::ScatterGlobal(cta, tile, items, d_out, guarded_elements);
		}


	};


	/**
	 * Terminate iteration
	 */
	template <int DUMMY>
	struct IterateTileElements<LOAD_VEC_SIZE, DUMMY>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile) {}

		// ComputeRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile) {}

		// ScatterRanked
		template <bool BANK_PADDING, typename Cta, typename Tile, typename T>
		static __device__ __forceinline__ void ScatterRanked(Cta *cta, Tile *tile, T items[LOAD_VEC_SIZE]) {}

		// GatherShared
		template <bool BANK_PADDING, typename Cta, typename Tile, typename T>
		static __device__ __forceinline__ void GatherShared(Cta *cta, Tile *tile, T items[LOAD_VEC_SIZE]) {}

		// DecodeBinOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeBinOffsets(Cta *cta, Tile *tile) {}

		// ScatterGlobal
		template <typename Cta, typename Tile, typename T>
		static __device__ __forceinline__ void ScatterGlobal(Cta *cta, Tile *tile, T items[LOAD_VEC_SIZE], T *d_out, const SizeT &guarded_elements) {}
	};


	//---------------------------------------------------------------------
	// Tile Internal Methods
	//---------------------------------------------------------------------


	template <typename Cta>
	__device__ __forceinline__ void RakingScan(Cta *cta)
	{

		if ((KernelPolicy::THREADS == RAKING_THREADS) || (threadIdx.x < RAKING_THREADS)) {

			// Upsweep reduce
			int raking_partial = util::reduction::SerialReduce<KernelPolicy::PADDED_RAKING_SEG>::Invoke(
				cta->raking_segment);

			// Warpscan

			int partial = raking_partial;
			cta->warpscan[0] = partial;

			cta->warpscan[0] = partial =
				partial + cta->warpscan[0 - 1];
			cta->warpscan[0] = partial =
				partial + cta->warpscan[0 - 2];
			cta->warpscan[0] = partial =
				partial + cta->warpscan[0 - 4];
			cta->warpscan[0] = partial =
				partial + cta->warpscan[0 - 8];
			cta->warpscan[0] = partial =
				partial + cta->warpscan[0 - 16];

			// Restricted barrier
			if (KernelPolicy::RAKING_WARPS > 1) util::BAR(RAKING_THREADS);

			// Scan across warpscan totals
			int warpscan_totals = 0;

			#pragma unroll
			for (int WARP = 0; WARP < KernelPolicy::RAKING_WARPS; WARP++) {

				// Add totals from all previous warpscans into our partial
				int warpscan_total = cta->smem_storage.warpscan[((WARP + 1) * (WARP_THREADS * 2)) - 1];
				if (cta->warp_id == (WARP * 32)) {
					partial += warpscan_totals;
				}

				// Increment warpscan totals
				warpscan_totals += warpscan_total;
			}

			// Add lower totals from all warpscans into partial's upper
			partial = util::SHL_ADD(warpscan_totals, 16, partial);

			// Downsweep scan with exclusive partial
			util::scan::SerialScan<KernelPolicy::PADDED_RAKING_SEG>::Invoke(
				cta->raking_segment,
				partial - raking_partial);
		}
	}


	/**
	 * Scan Tile
	 */
	template <int CURRENT_BIT, typename Cta>
	__device__ __forceinline__ void ScanTile(Cta *cta)
	{
		// Initialize raking lanes
		if ((KernelPolicy::THREADS == RAKING_THREADS) || (threadIdx.x < RAKING_THREADS)) {

			#pragma unroll
			for (int ELEMENT = 0; ELEMENT < KernelPolicy::PADDED_RAKING_SEG; ELEMENT++) {
				cta->raking_segment[ELEMENT] = 0;
			}
		}

		__syncthreads();

		// Decode bins and update counters
		IterateTileElements<0>::DecodeKeys(cta, this);

		__syncthreads();

		// Raking multi-scan
		RakingScan(cta);

		__syncthreads();

		// Update carry
		if ((KernelPolicy::THREADS == KernelPolicy::BINS) || (threadIdx.x < KernelPolicy::BINS)) {

			unsigned short bin_exclusive = cta->bin_counter[0];
			unsigned short bin_inclusive = cta->bin_counter[KernelPolicy::THREADS * 2];

			cta->my_bin_carry -= bin_exclusive;
			cta->smem_storage.bin_carry[threadIdx.x] = cta->my_bin_carry;
			cta->my_bin_carry += bin_inclusive;
		}

		// Extract the local ranks of each key
		IterateTileElements<0>::ComputeRanks(cta, this);
	}


	/**
	 * Load tile of keys
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		SizeT pack_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		KeyVectorType *vectors = (KeyVectorType *) keys;

		if (guarded_elements >= KernelPolicy::TILE_ELEMENTS) {

			// Unguarded loads through tex
			#pragma unroll
			for (int PACK = 0; PACK < PACKS_PER_LOAD; PACK++) {

				vectors[PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						KeysTex<KeyVectorType>::ref1 :
						KeysTex<KeyVectorType>::ref0,
					pack_offset + (threadIdx.x * PACKS_PER_LOAD) + PACK);
			}

		} else {

			// Guarded loads with default assignment of -1 to out-of-bound keys
			util::io::LoadTile<
				0,									// log loads per tile
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
					(KeyType (*)[LOAD_VEC_SIZE]) keys,
					(Cta::FLOP_TURN) ?
						cta->d_keys1 :
						cta->d_keys0,
					(pack_offset * KernelPolicy::PACK_SIZE),
					guarded_elements,
					KeyType(-1));
		}
	}

	/**
	 * Load tile of values
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadValues(
		SizeT pack_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		ValueVectorType *vectors = (ValueVectorType *) values;

		if (guarded_elements >= KernelPolicy::TILE_ELEMENTS) {

			// Unguarded loads through tex
			#pragma unroll
			for (int PACK = 0; PACK < PACKS_PER_LOAD; PACK++) {

				vectors[PACK] = tex1Dfetch(
					(Cta::FLOP_TURN) ?
						ValuesTex<ValueVectorType>::ref1 :
						ValuesTex<ValueVectorType>::ref0,
					pack_offset + (threadIdx.x * PACKS_PER_LOAD) + PACK);
			}

		} else {

			// Guarded loads with default assignment of -1 to out-of-bound values
			util::io::LoadTile<
				0,									// log loads per tile
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
					(KeyType (*)[LOAD_VEC_SIZE]) values,
					(Cta::FLOP_TURN) ?
						cta->d_values1 :
						cta->d_values0,
					(pack_offset * KernelPolicy::PACK_SIZE),
					guarded_elements);
		}
	}


	/**
	 * Helper structure for processing values.  (Specialized for keys-only
	 * passes.)
	 */
	template <
		typename ValueType,
		bool PADDED_EXCHANGE,
		bool TRUCK = KernelPolicy::KEYS_ONLY>
	struct TruckValues
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT pack_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// do nothing
		}
	};

	/**
	 * Helper structure for processing values.  (Specialized for key-value
	 * passes.)
	 */
	template <
		typename ValueType,
		bool PADDED_EXCHANGE>
	struct TruckValues<ValueType, PADDED_EXCHANGE, false>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT pack_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load tile of values
			tile->LoadValues(pack_offset, guarded_elements, cta);

			__syncthreads();

			// Scatter values shared
			IterateTileElements<0>::template ScatterRanked<PADDED_EXCHANGE>(cta, tile, tile->values);

			__syncthreads();

			// Gather values shared
			IterateTileElements<0>::template GatherShared<PADDED_EXCHANGE>(cta, tile, tile->values);

			// Scatter to global
			IterateTileElements<0>::ScatterGlobal(
				cta,
				tile,
				tile->values,
				(Cta::FLOP_TURN) ?
					cta->d_values0 :
					cta->d_values1,
				guarded_elements);
		}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------


	/**
	 * Loads, decodes, and scatters a tile into global partitions
	 */
	template <typename Cta>
	__device__ __forceinline__ void Partition(
		SizeT pack_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		// Whether or not to insert padding for exchanging keys
		const bool PADDED_EXCHANGE = true;

		// Load tile of keys
		LoadKeys(pack_offset, guarded_elements, cta);

		// Scan tile (computing padded exchange offsets)
		ScanTile<KernelPolicy::CURRENT_BIT>(cta);

		__syncthreads();

		// Scatter keys shared
		IterateTileElements<0>::template ScatterRanked<PADDED_EXCHANGE>(cta, this, keys);

		__syncthreads();

		// Gather keys
		IterateTileElements<0>::template GatherShared<PADDED_EXCHANGE>(cta, this, keys);

		// Decode global scatter offsets
		IterateTileElements<0>::DecodeBinOffsets(cta, this);

		// Scatter to global
		IterateTileElements<0>::ScatterGlobal(
			cta,
			this,
			keys,
			(Cta::FLOP_TURN) ?
				cta->d_keys0 :
				cta->d_keys1,
			guarded_elements);

		TruckValues<ValueType, PADDED_EXCHANGE>::Invoke(
			pack_offset,
			guarded_elements,
			cta,
			this);
	}

};


} // namespace downsweep
} // namespace partition
} // namespace b40c

