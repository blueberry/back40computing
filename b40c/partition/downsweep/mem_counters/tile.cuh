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

		LOG_SUB_COUNTERS			= KernelPolicy::LOG_SCAN_BINS - LOG_SCAN_LANES,
		SUB_COUNTERS				= 1 << LOG_SUB_COUNTERS,

		LOG_SIZEOF_SUB_COUNTER		= util::Log2<sizeof(int) / sizeof(short)>::VALUE,		// two bytes per sub-counter

		LOG_PACKS_PER_LOAD			= KernelPolicy::LOG_LOAD_VEC_SIZE - KernelPolicy::LOG_PACK_SIZE,
		PACKS_PER_LOAD				= 1 << LOG_PACKS_PER_LOAD,

		LANE_ROWS_PER_LOAD 			= KernelPolicy::RakingGrid::ROWS_PER_LANE / KernelPolicy::LOADS_PER_TILE,
		LANE_STRIDE_PER_LOAD 		= KernelPolicy::RakingGrid::PADDED_PARTIALS_PER_ROW * LANE_ROWS_PER_LOAD,

		INVALID_BIN					= -1,

		LOG_RAKING_THREADS 			= KernelPolicy::RakingGrid::LOG_RAKING_THREADS,
		RAKING_THREADS 				= 1 << LOG_RAKING_THREADS,

		LOG_WARPSCAN_THREADS		= B40C_LOG_WARP_THREADS(CUDA_ARCH),
		WARPSCAN_THREADS 			= 1 << LOG_WARPSCAN_THREADS,

	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::RakingGrid::LanePartial LanePartial;

	// The keys (and values) this thread will read this tile
	KeyType 	keys[LOADS_PER_TILE][LOAD_VEC_SIZE];
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];

	short 		prefix[LOADS_PER_TILE][LOAD_VEC_SIZE];
	short*		counter[LOADS_PER_TILE][LOAD_VEC_SIZE];

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
		static __device__ __forceinline__ void DecodeKeys(Cta *cta,	Tile *tile, LanePartial lane_partial)
		{

			int sub_counter = util::BFE(
				tile->keys[LOAD][VEC],
				CURRENT_BIT + KernelPolicy::LOG_SCAN_BINS - 1,
				1);

			int lane = util::BFE(
				tile->keys[LOAD][VEC],
				CURRENT_BIT,
				KernelPolicy::LOG_SCAN_BINS - 1);

			tile->counter[LOAD][VEC] = ((short *) lane_partial[lane]) + sub_counter;

			// Load thread-exclusive prefix
			tile->prefix[LOAD][VEC] = *tile->counter[LOAD][VEC];

			// Store inclusive prefix
			*tile->counter[LOAD][VEC] = tile->prefix[LOAD][VEC] + 4;

			// Next vector element
			IterateTileElements<
				CURRENT_BIT,
				PADDED_EXCHANGE,
				LOAD,
				VEC + 1>::DecodeKeys(cta, tile, lane_partial);
		}

		// ComputeRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// Add in CTA exclusive prefix
			tile->prefix[LOAD][VEC] += *tile->counter[LOAD][VEC];

			if (PADDED_EXCHANGE) {
				tile->prefix[LOAD][VEC] += (tile->prefix[LOAD][VEC] >> 7) << 2;
			}
/*
			printf("Thread(%d) key(%d) bin(%d) prefix(%d)\n",
				threadIdx.x,
				tile->keys[LOAD][VEC],
				util::BFE(tile->keys[LOAD][VEC], CURRENT_BIT, KernelPolicy::LOG_SCAN_BINS),
				tile->prefix[LOAD][VEC]);
*/
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
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile, LanePartial lane_partial)
		{
			// First vector element, next load
			IterateTileElements<
				CURRENT_BIT,
				PADDED_EXCHANGE,
				LOAD + 1,
				0>::DecodeKeys(cta, tile, lane_partial);
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
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile, LanePartial lane_partial) {}

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
		LanePartial lane_partial = (LanePartial)
			(cta->smem_storage.raking_lanes + threadIdx.x + (threadIdx.x >> KernelPolicy::RakingGrid::LOG_PARTIALS_PER_ROW));

		// Initialize lanes
		#pragma unroll
		for (int LANE = 0; LANE < SCAN_LANES; LANE++) {
			lane_partial[LANE][0] = 0;
		}

		// Decode bins and place keys into grid
		IterateTileElements<CURRENT_BIT, PADDED_EXCHANGE, 0, 0>::DecodeKeys(cta, this, lane_partial);

		__syncthreads();

		// Raking multi-scan

		int tid = threadIdx.x & 31;
		int warp = threadIdx.x >> 5;
		int other_warp = warp ^ 1;
		volatile int *warpscan = cta->smem_storage.warpscan[warp][1];
		volatile int *other_warpscan = cta->smem_storage.warpscan[other_warp][1];

		int *raking_segment =
			cta->smem_storage.raking_lanes +
			(threadIdx.x << KernelPolicy::RakingGrid::LOG_PARTIALS_PER_SEG) +
			(threadIdx.x >> KernelPolicy::RakingGrid::LOG_SEGS_PER_ROW);

		if ((KernelPolicy::THREADS == RAKING_THREADS) || (threadIdx.x < RAKING_THREADS)) {

			// Upsweep scan
			int raking_partial = util::reduction::SerialReduce<KernelPolicy::RakingGrid::PARTIALS_PER_SEG>::Invoke(
				raking_segment);

			// Warpscan
			int partial = raking_partial;
			warpscan[tid] = partial;

			warpscan[tid] = partial =
				partial + warpscan[tid - 1];
			warpscan[tid] = partial =
				partial + warpscan[tid - 2];
			warpscan[tid] = partial =
				partial + warpscan[tid - 4];
			warpscan[tid] = partial =
				partial + warpscan[tid - 8];
			warpscan[tid] = partial =
				partial + warpscan[tid - 16];

			// Restricted barrier
			util::BAR(RAKING_THREADS);

			// grab own total
			int total = warpscan[B40C_WARP_THREADS(CUDA_ARCH) - 1];

			// Add lower into upper
			partial = util::SHL_ADD_C(total, 16, partial);

			// Grab other warp's total
			int other_total = other_warpscan[B40C_WARP_THREADS(CUDA_ARCH) - 1];
			int shifted_other_total = other_total << 16;
			if (warp) shifted_other_total += other_total;
			partial += shifted_other_total;

			util::scan::SerialScan<KernelPolicy::RakingGrid::PARTIALS_PER_SEG>::Invoke(
				raking_segment,
				partial - raking_partial);

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

			linear_keys[ELEMENT] = *(cta->smem_storage.key_exchange + (ELEMENT * KernelPolicy::THREADS) + threadIdx.x);


/*
			printf("Thread(%d) key(%d)\n",
				threadIdx.x,
				linear_keys[ELEMENT]);
*/

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
			*(cta->d_out_keys + (ELEMENT * KernelPolicy::THREADS) + threadIdx.x) = linear_keys[ELEMENT];

/*
			int bin_carry = cta->smem_storage.bin_carry[tile->bins[ELEMENT]];
			int tile_element = threadIdx.x + (ELEMENT * KernelPolicy::THREADS);

			printf("\tTid %d scattering key[%d](%d) with bin_carry(%d) to offset %d\n",
				threadIdx.x,
				ELEMENT,
				linear_keys[ELEMENT],
				bin_carry,
				threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + bin_carry);

			if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					linear_keys[ELEMENT],
					cta->d_out_keys + threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + bin_carry);
			}
*/
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

			// Scatter keys to smem by local rank (strided)
			char *exchange = (char *) cta->smem_storage.key_exchange;

			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {
				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {
					KeyType* loc = (KeyType*)(exchange + tile->prefix[LOAD][VEC]);
					*loc = tile->keys[LOAD][VEC];
				}
			}

			__syncthreads();

			int gather_base = (threadIdx.x * LOAD_VEC_SIZE) +
				((threadIdx.x * LOAD_VEC_SIZE) >> B40C_MAX(5, KernelPolicy::LOG_LOAD_VEC_SIZE));

			// Gather keys from smem (strided)
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {

				const int LOAD_IDX = LOAD * LOAD_VEC_SIZE * KernelPolicy::THREADS;
				const int LOAD_OFFSET = LOAD_IDX + (LOAD_IDX >> B40C_MAX(5, KernelPolicy::LOG_LOAD_VEC_SIZE));

				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {

					tile->keys[LOAD][VEC] = *(cta->smem_storage.key_exchange +
						LOAD_OFFSET + VEC + gather_base);
				}
			}

			// Scan tile (no padded exchange offsets)
			tile->template ScanTile<KernelPolicy::CURRENT_BIT + KernelPolicy::LOG_SCAN_BINS, false>(cta);

			// Scatter keys to smem by local rank (strided)
			#pragma unroll
			for (int LOAD = 0; LOAD < KernelPolicy::LOADS_PER_TILE; LOAD++) {
				#pragma unroll
				for (int VEC = 0; VEC < LOAD_VEC_SIZE; VEC++) {
					KeyType* loc = (KeyType*)(exchange + tile->prefix[LOAD][VEC]);
					*loc = tile->keys[LOAD][VEC];
				}
			}

			__syncthreads();

			// Gather keys linearly from smem (also saves off bin in/exclusives)
			IterateElements<0>::GatherDecodeKeys(cta, tile);
/*
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
*/
			// Scatter keys to global bin partitions
			IterateElements<0>::ScatterKeysToGlobal(cta, tile, guarded_elements);

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

