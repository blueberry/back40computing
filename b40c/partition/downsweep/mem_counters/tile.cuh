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

		LOG_PACKS_PER_LOAD			= KernelPolicy::LOG_LOAD_VEC_SIZE - KernelPolicy::LOG_PACK_SIZE,
		PACKS_PER_LOAD				= 1 << LOG_PACKS_PER_LOAD,

		LOG_RAKING_THREADS 			= KernelPolicy::RakingGrid::LOG_RAKING_THREADS,
		RAKING_THREADS 				= 1 << LOG_RAKING_THREADS,

		RAKING_LANES				= KernelPolicy::RAKING_LANES,

		WARP_THREADS				= B40C_WARP_THREADS(KernelPolicy::CUDA_ARCH),
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// The keys (and values) this thread will read this tile
	KeyType 	keys[LOAD_VEC_SIZE];
	ValueType 	values[LOAD_VEC_SIZE];

	short 		prefixes[LOAD_VEC_SIZE];
	short*		counters[LOAD_VEC_SIZE];
	int 		bins[LOAD_VEC_SIZE];


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

			tile->counters[VEC] = ((short *) cta->counters) + (lane * KernelPolicy::THREADS * 2) + sub_counter;

			// Load thread-exclusive prefix
			tile->prefixes[VEC] = *tile->counters[VEC];

			// Store inclusive prefix
			*tile->counters[VEC] = tile->prefixes[VEC] + 4;

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
			if (BANK_PADDING) tile->prefixes[VEC] += (tile->prefixes[VEC] >> 7) << 2;

			// Next vector element
			IterateTileElements<VEC + 1>::template ComputeRanks<BANK_PADDING>(cta, tile);
		}


		// ScatterRanked
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScatterRanked(Cta *cta, Tile *tile)
		{
			KeyType *dest = (KeyType *) ((char *) cta->smem_storage.key_exchange + tile->prefixes[VEC]);
			*dest = tile->keys[VEC];

			// Next vector element
			IterateTileElements<VEC + 1>::ScatterRanked(cta, tile);
		}


		// GatherScatterKeys
		template <bool BANK_PADDING, typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherScatterKeys(
			Cta *cta,
			Tile *tile,
			const SizeT &guarded_elements)
		{
			const int LOAD_OFFSET = (BANK_PADDING) ?
				(VEC * KernelPolicy::THREADS) + ((VEC * KernelPolicy::THREADS) >> 5) :
				(VEC * KernelPolicy::THREADS);

			// Gather and decode key
			KeyType *base_gather_offset 	= cta->smem_storage.key_exchange + threadIdx.x;

			// Add padding
			if (BANK_PADDING) base_gather_offset += threadIdx.x >> 5;

			KeyType key 					= base_gather_offset[LOAD_OFFSET];
			int bin 						= util::BFE(key, KernelPolicy::CURRENT_BIT, KernelPolicy::LOG_BINS);

			// Lookup bin carry
			int bin_carry = cta->smem_storage.bin_carry[bin];

			// Distribute
			int tile_element = threadIdx.x + (VEC * KernelPolicy::THREADS);

			if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

				util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
					key,
					cta->d_out_keys + threadIdx.x + (KernelPolicy::THREADS * VEC) + bin_carry);
			}

			// Next vector element
			IterateTileElements<VEC + 1>::template GatherScatterKeys<BANK_PADDING>(cta, tile, guarded_elements);
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

		// GatherScatterKeys
		template <bool BANK_PADDING, typename Cta, typename Tile>
		static __device__ __forceinline__ void GatherScatterKeys(Cta *cta, Tile *tile, const SizeT &guarded_elements) {}
	};


	//---------------------------------------------------------------------
	// Iterate raking lanes
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int LANE, int DUMMY = 0>
	struct IterateRakingLanes
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpsweepLanes(Cta *cta, Tile *tile)
		{
			*cta->lane_partial[LANE] = util::reduction::SerialReduce<4>::Invoke(
				cta->smem_storage.paired_counters_32[LANE][threadIdx.x]);

			// Next lane
			IterateRakingLanes<LANE + 1>::UpsweepLanes(cta, tile);
		}

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DownsweepLanes(Cta *cta, Tile *tile)
		{
			util::scan::SerialScan<4>::Invoke(
				cta->smem_storage.paired_counters_32[LANE][threadIdx.x],
				*cta->lane_partial[LANE]);

			if (LANE < RAKING_LANES - 1) __threadfence_block();

			// Next lane
			IterateRakingLanes<LANE + 1>::DownsweepLanes(cta, tile);
		}

	};

	template <int DUMMY>
	struct IterateRakingLanes<RAKING_LANES, DUMMY>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpsweepLanes(Cta *cta, Tile *tile) {}

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DownsweepLanes(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// Tile Internal Methods
	//---------------------------------------------------------------------


	template <typename Cta>
	__device__ __forceinline__ void RakingScan64(Cta *cta)
	{

		if ((KernelPolicy::THREADS == RAKING_THREADS) || (threadIdx.x < RAKING_THREADS)) {

			volatile int * warpscan(cta->smem_storage.warpscan[threadIdx.x >> 5] + (WARP_THREADS / 2) + (threadIdx.x & 31));

			// Upsweep reduce
			int raking_partial = util::reduction::SerialReduce<KernelPolicy::RakingGrid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment);

			// Warpscan
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
			util::BAR(RAKING_THREADS);

			int lower_total = cta->smem_storage.warpscan[0][(WARP_THREADS * 3 / 2) - 1];
			int upper_total = cta->smem_storage.warpscan[1][(WARP_THREADS * 3 / 2) - 1];

			// Add lower's lower into upper
			partial = util::SHL_ADD(lower_total, 16, partial);

			// Add upper's lower into upper
			partial = util::SHL_ADD(upper_total, 16, partial);

			// Add lower into upper
			if (threadIdx.x >> 5) partial += lower_total;

			// Downsweep scan with exclusive partial
			util::scan::SerialScan<KernelPolicy::RakingGrid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment,
				partial - raking_partial);

			// take out byte-multiplier
			partial >>= 2;

			// Store off bin inclusives
			const int LOG_RAKING_THREADS_PER_COUNTER_LANE = LOG_RAKING_THREADS - KernelPolicy::LOG_SCAN_LANES;
			const int BIN_MASK = (1 << LOG_RAKING_THREADS_PER_COUNTER_LANE) - 1;
			int low_bin = threadIdx.x >> LOG_RAKING_THREADS_PER_COUNTER_LANE;

			if (threadIdx.x & BIN_MASK) {
				cta->smem_storage.bin_prefixes[1 + low_bin] = partial & 0x0000ffff;
				cta->smem_storage.bin_prefixes[1 + (KernelPolicy::BINS / 2) + low_bin] = partial >> 16;
			}
		}
	}

	template <typename Cta>
	__device__ __forceinline__ void RakingScan32(Cta *cta)
	{
		if ((KernelPolicy::THREADS == RAKING_THREADS) || (threadIdx.x < RAKING_THREADS)) {

			volatile int *warpscan(cta->smem_storage.warpscan[0] + (WARP_THREADS / 2) + threadIdx.x);

			// Upsweep reduce
			int raking_partial = util::reduction::SerialReduce<KernelPolicy::RakingGrid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment);

			// Warpscan
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

			// grab own total
			int total = cta->smem_storage.warpscan[0][(WARP_THREADS * 3 / 2) - 1];

			// Add lower total into upper
			partial = util::SHL_ADD(total, 16, partial);

			// Downsweep scan with exclusive partial
			util::scan::SerialScan<KernelPolicy::RakingGrid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment,
				partial - raking_partial);

			// take out byte-multiplier
			partial >>= 2;

			// Store off bin inclusives
			const int LOG_RAKING_THREADS_PER_COUNTER_LANE = LOG_RAKING_THREADS - KernelPolicy::LOG_SCAN_LANES;
			const int BIN_MASK = (1 << LOG_RAKING_THREADS_PER_COUNTER_LANE) - 1;
			int low_bin = threadIdx.x >> LOG_RAKING_THREADS_PER_COUNTER_LANE;

			if (threadIdx.x & BIN_MASK) {
				cta->smem_storage.bin_prefixes[1 + low_bin] = partial & 0x0000ffff;
				cta->smem_storage.bin_prefixes[1 + (KernelPolicy::BINS / 2) + low_bin] = partial >> 16;
			}
		}
	}


	/**
	 * Scan Tile
	 */
	template <int CURRENT_BIT, int PADDED_EXCHANGE, typename Cta>
	__device__ __forceinline__ void ScanTile(Cta *cta)
	{
		// Initialize lanes
		#pragma unroll
		for (int LANE = 0; LANE < KernelPolicy::SCAN_LANES; LANE++) {
			cta->counters[LANE * KernelPolicy::THREADS] = 0;
		}

		// Decode bins and place keys into grid
		IterateTileElements<0>::DecodeKeys(cta, this);

		__syncthreads();

		// Upsweep reduce
		IterateRakingLanes<0>::UpsweepLanes(cta, this);

		__syncthreads();

		// Raking multi-scan
		RakingScan64(cta);

		__syncthreads();

		// Update carry
		if ((KernelPolicy::THREADS == KernelPolicy::BINS) || (threadIdx.x < KernelPolicy::BINS)) {

			int bin_inclusive = cta->smem_storage.bin_prefixes[1 + threadIdx.x];
			int bin_exclusive = cta->smem_storage.bin_prefixes[threadIdx.x];

			cta->my_bin_carry -= bin_exclusive;
			cta->smem_storage.bin_carry[threadIdx.x] = cta->my_bin_carry;
			cta->my_bin_carry += bin_inclusive;
		}

		// Downsweep scan
		IterateRakingLanes<0>::DownsweepLanes(cta, this);

		__syncthreads();

		// Extract the local ranks of each key
		IterateTileElements<0>::template ComputeRanks<false>(cta, this);
	}


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
			VectorType *vectors = (VectorType *) tile->keys;

			#pragma unroll
			for (int PACK = 0; PACK < PACKS_PER_LOAD; PACK++) {

				vectors[PACK] = tex1Dfetch(
					KeysTex<VectorType>::ref,
					pack_offset + (threadIdx.x * PACKS_PER_LOAD) + PACK);
			}

			// Scan tile (computing padded exchange offsets)
			tile->template ScanTile<KernelPolicy::CURRENT_BIT, true>(cta);

			__syncthreads();

			// Scatter keys shared
			IterateTileElements<0>::ScatterRanked(cta, tile);

			__syncthreads();

			// Gather keys and scatter to global
			IterateTileElements<0>::template GatherScatterKeys<false>(cta, tile, guarded_elements);

			__syncthreads();
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

