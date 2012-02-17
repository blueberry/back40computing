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

	typedef DerivedTile Dispatch;

	enum {
		LOAD_VEC_SIZE 				= KernelPolicy::LOAD_VEC_SIZE,

		LOG_PACKS_PER_LOAD			= KernelPolicy::LOG_LOAD_VEC_SIZE - KernelPolicy::LOG_PACK_SIZE,
		PACKS_PER_LOAD				= 1 << LOG_PACKS_PER_LOAD,

		LOG_RAKING_THREADS 			= KernelPolicy::LOG_RAKING_THREADS,
		RAKING_THREADS 				= 1 << LOG_RAKING_THREADS,

		WARP_THREADS				= B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH),
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// The keys (and values) this thread will read this tile
	KeyType 			keys[LOAD_VEC_SIZE];
	ValueType 			values[LOAD_VEC_SIZE];
	unsigned short 		prefixes[LOAD_VEC_SIZE];
	unsigned short*		counters[LOAD_VEC_SIZE];


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

			tile->counters[VEC] = cta->counters + (lane * KernelPolicy::THREADS * 2) + sub_counter;

			// Load thread-exclusive prefix
			tile->prefixes[VEC] = *tile->counters[VEC];

			// Store inclusive prefix
			*tile->counters[VEC] = tile->prefixes[VEC] + 1;

			// Next vector element
			IterateTileElements<VEC + 1>::DecodeKeys(cta, tile);
		}


		// ComputeRanks
		template <bool BANK_PADDING, typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// Add in CTA exclusive prefix
			tile->prefixes[VEC] += *tile->counters[VEC];

			// Add in padding
			if (BANK_PADDING) tile->prefixes[VEC] += (tile->prefixes[VEC] >> 5);

			// Next vector element
			IterateTileElements<VEC + 1>::template ComputeRanks<BANK_PADDING>(cta, tile);
		}


		// ScatterRanked
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterRanked(Cta *cta, Tile *tile)
		{
			cta->smem_storage.key_exchange[tile->prefixes[VEC]] = tile->keys[VEC];

			// Next vector element
			IterateTileElements<VEC + 1>::ScatterRanked(cta, tile);
		}

		// GatherShared
		template <bool BANK_PADDING, typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherShared(Cta *cta, Tile *tile)
		{
			const int LOAD_OFFSET = (BANK_PADDING) ?
				(VEC * KernelPolicy::THREADS) + ((VEC * KernelPolicy::THREADS) >> 5) :
				(VEC * KernelPolicy::THREADS);

			// Gather and decode key
			KeyType *base_gather_offset = cta->smem_storage.key_exchange + threadIdx.x;

			// Add padding
			if (BANK_PADDING) base_gather_offset += (threadIdx.x >> 5);

			tile->keys[VEC] = base_gather_offset[LOAD_OFFSET];

			// Next vector element
			IterateTileElements<VEC + 1>::template GatherShared<BANK_PADDING>(cta, tile);
		}

		// ScatterGlobal
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterGlobal(
			Cta *cta,
			Tile *tile,
			const SizeT &guarded_elements)
		{
			int bin = util::BFE(tile->keys[VEC], KernelPolicy::CURRENT_BIT, KernelPolicy::LOG_BINS);

			// Lookup bin carry
			int bin_carry = cta->smem_storage.bin_carry[bin];

			// Distribute
			int tile_element = threadIdx.x + (VEC * KernelPolicy::THREADS);

			if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				KeyType *d_out = (Cta::FLOP_TURN) ?
						cta->d_keys0 :
						cta->d_keys1;

				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					tile->keys[VEC],
					d_out + threadIdx.x + (KernelPolicy::THREADS * VEC) + bin_carry);
			}

			// Next vector element
			IterateTileElements<VEC + 1>::ScatterGlobal(cta, tile, guarded_elements);
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
		template <bool BANK_PADDING, typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile) {}

		// ScatterRanked
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterRanked(Cta *cta, Tile *tile) {}

		// GatherShared
		template <bool BANK_PADDING, typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherShared(Cta *cta, Tile *tile) {}

		// ScatterGlobal
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterGlobal(Cta *cta, Tile *tile, const SizeT &guarded_elements) {}
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
			int warp_id = (threadIdx.x & (~31));
			volatile int *warpscan = cta->smem_storage.warpscan + 32 + threadIdx.x + warp_id;

			int partial = raking_partial;
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

			// Restricted barrier
			if (KernelPolicy::RAKING_WARPS > 1) util::BAR(RAKING_THREADS);

			// Scan across warpscan totals
			int warpscan_totals = 0;

			#pragma unroll
			for (int WARP = 0; WARP < KernelPolicy::RAKING_WARPS; WARP++) {

				// Add totals from all previous warpscans into our partial
				int warpscan_total = cta->smem_storage.warpscan[((WARP + 1) * (WARP_THREADS * 2)) - 1];
				if (warp_id == (WARP * 32)) {
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
	 * Load tile of keys
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		SizeT pack_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		// Load keys
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
	 * Scan Tile
	 */
	template <int CURRENT_BIT, int PADDED_EXCHANGE, typename Cta>
	__device__ __forceinline__ void ScanTile(Cta *cta)
	{
		// Initialize raking lanes
		#pragma unroll
		for (int LANE = 0; LANE < KernelPolicy::SCAN_LANES + 1; LANE++) {
			int *counter = (int *) cta->counters;
			counter[LANE * KernelPolicy::THREADS] = 0;
		}

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
		IterateTileElements<0>::template ComputeRanks<PADDED_EXCHANGE>(cta, this);
	}


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

		__syncthreads();

		// Scan tile (computing padded exchange offsets)
		ScanTile<KernelPolicy::CURRENT_BIT, PADDED_EXCHANGE>(cta);

		__syncthreads();

		// Scatter keys shared
		IterateTileElements<0>::ScatterRanked(cta, this);

		__syncthreads();

		// Gather keys
		IterateTileElements<0>::template GatherShared<PADDED_EXCHANGE>(cta, this);

		// Scatter to global
		IterateTileElements<0>::ScatterGlobal(cta, this, guarded_elements);
	}

};


} // namespace downsweep
} // namespace partition
} // namespace b40c

