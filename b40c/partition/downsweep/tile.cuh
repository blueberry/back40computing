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
template <typename KeyType>
struct KeysTex
{
	static texture<KeyType, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename KeyType>
texture<KeyType, cudaTextureType1D, cudaReadModeElementType> KeysTex<KeyType>::ref;



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

	typedef DerivedTile Dispatch;

	enum {
		LOAD_VEC_SIZE 				= KernelPolicy::LOAD_VEC_SIZE,
		LOADS_PER_TILE 				= KernelPolicy::LOADS_PER_TILE,
		TILE_ELEMENTS_PER_THREAD 	= KernelPolicy::TILE_ELEMENTS_PER_THREAD,

		LOG_SCAN_LANES				= KernelPolicy::LOG_SCAN_LANES_PER_TILE,
		SCAN_LANES					= KernelPolicy::SCAN_LANES_PER_TILE,

		LOG_PACKS_PER_LOAD			= KernelPolicy::LOG_LOAD_VEC_SIZE - KernelPolicy::LOG_PACK_SIZE,
		PACKS_PER_LOAD				= 1 << LOG_PACKS_PER_LOAD,

		LANE_ROWS_PER_LOAD 			= KernelPolicy::ByteGrid::ROWS_PER_LANE / KernelPolicy::LOADS_PER_TILE,
		LANE_STRIDE_PER_LOAD 		= KernelPolicy::ByteGrid::PADDED_PARTIALS_PER_ROW * LANE_ROWS_PER_LOAD,

		INVALID_BIN					= -1,

		LOG_RAKING_THREADS 			= KernelPolicy::ByteGrid::LOG_RAKING_THREADS,
		RAKING_THREADS 				= 1 << LOG_RAKING_THREADS,

		LOG_WARPSCAN_THREADS		= B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPSCAN_THREADS 			= 1 << LOG_WARPSCAN_THREADS,

	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------


	// The keys (and values) this thread will read this tile
	KeyType 	keys[LOADS_PER_TILE][LOAD_VEC_SIZE];
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];

	short 		prefix[LOADS_PER_TILE][LOAD_VEC_SIZE];
	int		 	sub_counter[LOADS_PER_TILE][LOAD_VEC_SIZE];
	int		 	lane[LOADS_PER_TILE][LOAD_VEC_SIZE];




	//---------------------------------------------------------------------
	// IterateTileElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int CURRENT_BIT, bool PADDED_EXCHANGE, int LOAD, int VEC>
	struct IterateTileElements
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta,	Tile *tile)
		{
			tile->sub_counter[LOAD][VEC] = util::BFE(
				tile->keys[LOAD][VEC],
				CURRENT_BIT,
				1);

			tile->lane[LOAD][VEC] = util::BFE(
				tile->keys[LOAD][VEC],
				CURRENT_BIT + 1,
				KernelPolicy::LOG_SCAN_BINS - 1);

			// Load exclusive prefix
			tile->prefix[LOAD][VEC] =
				cta->smem_storage.short_counters[tile->lane[LOAD][VEC]][threadIdx.x][tile->sub_counter[LOAD][VEC]];

			printf("Thread(%d) key(%d) sub_counter(%d) lane(%d) prefix(%d)\n",
				threadIdx.x,
				tile->keys[LOAD][VEC],
				tile->sub_counter[LOAD][VEC],
				tile->lane[LOAD][VEC],
				tile->prefix[LOAD][VEC]);

			// Store inclusive prefix
			cta->smem_storage.short_counters[tile->lane[LOAD][VEC]][threadIdx.x][tile->sub_counter[LOAD][VEC]] =
				tile->prefix[LOAD][VEC] + 1;

			// Next vector element
			IterateTileElements<
				CURRENT_BIT,
				PADDED_EXCHANGE,
				LOAD,
				VEC + 1>::DecodeKeys(cta, tile);
		}

		// ComputeRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// ...


			// Next vector element
			IterateTileElements<
				CURRENT_BIT,
				PADDED_EXCHANGE,
				LOAD,
				VEC + 1>::ComputeRanks(cta, tile);
		}
	};



	/**
	 * IterateTileElements next load
	 */
	template <int CURRENT_BIT, bool PADDED_EXCHANGE, int LOAD>
	struct IterateTileElements<CURRENT_BIT, PADDED_EXCHANGE, LOAD, LOAD_VEC_SIZE>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			// First vector element, next load
			IterateTileElements<
				CURRENT_BIT,
				PADDED_EXCHANGE,
				LOAD + 1,
				0>::DecodeKeys(cta, tile);
		}

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// First vector element, next load
			IterateTileElements<
				CURRENT_BIT,
				PADDED_EXCHANGE,
				LOAD + 1,
				0>::ComputeRanks(cta, tile);
		}

	};

	/**
	 * Terminate iteration
	 */
	template <int CURRENT_BIT, bool PADDED_EXCHANGE>
	struct IterateTileElements<CURRENT_BIT, PADDED_EXCHANGE, LOADS_PER_TILE, 0>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile) {}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile) {}
	};



	//---------------------------------------------------------------------
	// Tile Internal Methods
	//---------------------------------------------------------------------


	/**
	 * Scan Tile
	 */
	template <int CURRENT_BIT, int PADDED_EXCHANGE, typename Cta>
	__device__ __forceinline__ void ScanTile(Cta *cta)
	{
		// Initialize lanes
		#pragma unroll
		for (int LANE = 0; LANE < SCAN_LANES; LANE++) {
			cta->smem_storage.int_counters[LANE][threadIdx.x] = 0;
		}

		// Decode bins and place keys into grid
		IterateTileElements<CURRENT_BIT, PADDED_EXCHANGE, 0, 0>::DecodeKeys(cta, this);

		// Raking multi-scan
		if (threadIdx.x < RAKING_THREADS) {

			// ...

			// Restricted barrier
			util::BAR(RAKING_THREADS);

			// ...

		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateTileElements<CURRENT_BIT, PADDED_EXCHANGE, 0, 0>::ComputeRanks(cta, this);

	}



	//---------------------------------------------------------------------
	// IterateElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next tile element
	 */
	template <int ELEMENT, int dummy = 0>
	struct IterateElements
	{
		// GatherDecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherDecodeKeys(Cta *cta, Tile *tile)
		{
			KeyType *linear_keys = (KeyType *) tile->keys;

			KeyType *gather = cta->smem_storage.key_exchange + (ELEMENT * KernelPolicy::THREADS) + threadIdx.x;

			linear_keys[ELEMENT] = *(gather);
/*
			KeyType next_key = *(gather + 1);

			tile->bins[ELEMENT] = util::BFE(
				linear_keys[ELEMENT],
				KernelPolicy::CURRENT_BIT,
				KernelPolicy::LOG_BINS);

			int2 item;	// (inclusive for bins[element], next bin)
			item.x = threadIdx.x + (ELEMENT * KernelPolicy::THREADS);
			item.y = ((ELEMENT == TILE_ELEMENTS_PER_THREAD - 1) && (threadIdx.x == KernelPolicy::THREADS - 1)) ?
				KernelPolicy::BINS :						// invalid bin
				util::BFE(
					next_key,
					KernelPolicy::CURRENT_BIT,
					KernelPolicy::LOG_BINS);

			if (tile->bins[ELEMENT] != item.y) {
				cta->smem_storage.bin_in_prefixes[tile->bins[ELEMENT]] = item;
			}
*/
			IterateElements<ELEMENT + 1>::GatherDecodeKeys(cta, tile);
		}

		// ScatterKeysToGlobal
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterKeysToGlobal(
			Cta *cta,
			Tile *tile,
			const SizeT &guarded_elements)
		{
			KeyType *linear_keys = (KeyType *) tile->keys;

			int bin_carry = cta->smem_storage.bin_carry[tile->bins[ELEMENT]];
			int tile_element = threadIdx.x + (ELEMENT * KernelPolicy::THREADS);
/*
			printf("\tTid %d scattering key[%d](%d) with bin_carry(%d) to offset %d\n",
				threadIdx.x,
				ELEMENT,
				linear_keys[ELEMENT],
				bin_carry,
				threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + bin_carry);
*/
			if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					linear_keys[ELEMENT],
					cta->d_out_keys + threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + bin_carry);
			}

			IterateElements<ELEMENT + 1>::ScatterKeysToGlobal(cta, tile, guarded_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateElements<TILE_ELEMENTS_PER_THREAD, dummy>
	{
		// GatherDecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherDecodeKeys(
			Cta *cta, Tile *tile) {}

		// ScatterKeysToGlobal
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterKeysToGlobal(
			Cta *cta, Tile *tile, const SizeT &guarded_elements) {}
	};



	//---------------------------------------------------------------------
	// Partition/scattering specializations
	//---------------------------------------------------------------------


	/**
	 * Specialized for two-phase scatter, keys-only
	 */
	template <ScatterStrategy SCATTER_STRATEGY>
	struct PartitionTile
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT pack_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load keys
			typedef typename util::VecType<KeyType, KernelPolicy::PACK_SIZE>::Type VectorType;
			VectorType (*vectors)[PACKS_PER_LOAD] = (VectorType (*)[PACKS_PER_LOAD]) tile->keys;

			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int PACK = 0; PACK < PACKS_PER_LOAD; PACK++) {

					vectors[LOAD][PACK] = tex1Dfetch(
						KeysTex<VectorType>::ref,
						pack_offset + (threadIdx.x * PACKS_PER_LOAD) + (LOAD * KernelPolicy::THREADS * PACKS_PER_LOAD) + PACK);
				}
			}

			// Scan tile (computing padded exchange offsets)
			tile->template ScanTile<KernelPolicy::CURRENT_BIT, true>(cta);

/*
			__syncthreads();

			// Scatter keys to smem by local rank (strided)
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					char * ptr = (char *) cta->smem_storage.key_exchange;
					KeyType * ptr_key = (KeyType *)(ptr + tile->local_ranks[LOAD][VEC]);

					*ptr_key = tile->keys[LOAD][VEC];
				}
			}

			__syncthreads();

			// Gather keys from smem (strided)
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					const int LOAD_IDX = LOAD * LOAD_VEC_SIZE * KernelPolicy::THREADS;
					const int LOAD_OFFSET = LOAD_IDX + (LOAD_IDX >> B40C_MAX(5, KernelPolicy::LOG_LOAD_VEC_SIZE));

					tile->keys[LOAD][VEC] = cta->smem_storage.key_exchange[
						(threadIdx.x * LOAD_VEC_SIZE) +
						((threadIdx.x * LOAD_VEC_SIZE) >> B40C_MAX(5, KernelPolicy::LOG_LOAD_VEC_SIZE)) +
						LOAD_OFFSET +
						VEC];
				}
			}

			__syncthreads();

			// Scan tile (no padded exchange offsets)
			tile->template ScanTile<KernelPolicy::CURRENT_BIT + KernelPolicy::LOG_SCAN_BINS, false>(cta);

			__syncthreads();

			// Scatter keys to smem by local rank (strided)
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					char * ptr = (char *) cta->smem_storage.key_exchange;
					KeyType * ptr_key = (KeyType *)(ptr + tile->local_ranks[LOAD][VEC]);

					*ptr_key = tile->keys[LOAD][VEC];
				}
			}

			__syncthreads();

			// Gather keys linearly from smem (also saves off bin in/exclusives)
			IterateElements<0>::GatherDecodeKeys(cta, tile);

			__syncthreads();

			if (threadIdx.x < KernelPolicy::BINS) {

				// Put exclusive count into corresponding bin
				int2 item = cta->smem_storage.bin_in_prefixes[threadIdx.x];
				int bin_inclusive = item.x + 1;
				cta->smem_storage.bin_ex_prefixes[item.y] = bin_inclusive;

				// Restricted barrier
				util::BAR(KernelPolicy::BINS);

				int bin_exclusive = cta->smem_storage.bin_ex_prefixes[threadIdx.x];

				cta->my_bin_carry -= bin_exclusive;
				cta->smem_storage.bin_carry[threadIdx.x] = cta->my_bin_carry;
				cta->my_bin_carry += bin_inclusive;

				item.x = -1;
				item.y = KernelPolicy::BINS;
				cta->smem_storage.bin_in_prefixes[threadIdx.x] = item;

			}

			__syncthreads();

			// Scatter keys to global bin partitions
			IterateElements<0>::ScatterKeysToGlobal(cta, tile, guarded_elements);
*/
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
		PartitionTile<KernelPolicy::SCATTER_STRATEGY>::Invoke(
			pack_offset,
			guarded_elements,
			cta,
			(Dispatch *) this);

	}

};


} // namespace downsweep
} // namespace partition
} // namespace b40c

