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
 * Templated texture reference for collision bitmask
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

	// For each load:
	// 		counts_nibbles contains the bin counts within nibbles ordered right to left
	// 		bins_nibbles contains the bin for each key within nibbles ordered right to left
	// 		load_prefix_bytes contains the exclusive scan for each key within nibbles ordered right to left

	int 		bins_nibbles[(LOAD_VEC_SIZE + 7) / 8][LOADS_PER_TILE];

	int 		counts_nibbles[SCAN_LANES / 2][LOADS_PER_TILE];
	int			counts_bytes[SCAN_LANES][LOADS_PER_TILE];

	int 		load_prefix_bytes[(LOAD_VEC_SIZE + 3) / 4][LOADS_PER_TILE];

	int 		warpscan_shorts[LOADS_PER_TILE][4];

	int 		local_ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];		// The local rank of each key
	SizeT 		scatter_offsets[LOADS_PER_TILE][LOAD_VEC_SIZE];	// The global rank of each key


	//---------------------------------------------------------------------
	// Abstract Interface
	//---------------------------------------------------------------------

	/**
	 * Returns whether or not the key is valid.
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ SizeT ValidElements(Cta *cta, const SizeT &guarded_elements)
	{
		return guarded_elements;
	}

	/**
	 * Returns the bin into which the specified key is to be placed.
	 *
	 * To be overloaded
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(KeyType key, Cta *cta);


	/**
	 * Returns whether or not the key is valid.
	 *
	 * To be overloaded.
	 */
	template <int LOAD, int VEC>
	__device__ __forceinline__ bool IsValid();


	/**
	 * Loads keys into the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				(KeyType (*)[KernelPolicy::LOAD_VEC_SIZE]) keys,
				cta->d_in_keys,
				cta_offset,
				guarded_elements);
	}


	/**
	 * Scatter keys from the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterKeys(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		// Scatter keys to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_keys,
				(KeyType (*)[1]) keys,
				(SizeT (*)[1]) scatter_offsets,
				guarded_elements);
	}


	/**
	 * Loads values into the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadValues(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		// Read values
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				(ValueType (*)[KernelPolicy::LOAD_VEC_SIZE]) values,
				cta->d_in_values,
				cta_offset,
				guarded_elements);
	}


	/**
	 * Scatter values from the tile
	 *
	 * Can be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterValues(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		// Scatter values to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_values,
				(ValueType (*)[1]) values,
				(SizeT (*)[1]) scatter_offsets,
				guarded_elements);
	}


	//---------------------------------------------------------------------
	// Tile Methods
	//---------------------------------------------------------------------

	/**
	 * ExtractRanks
	 */
	template <int LOAD, int VEC, int REM = (VEC & 7)>
	struct ExtractRanks
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile) {}
	};


	/**
	 * ExtractRanks (VEC % 8 == 0)
	 */
	template <int LOAD, int VEC>
	struct ExtractRanks<LOAD, VEC, 0>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile)
		{
/*
			printf("\tTid(%d) Vec(%d) bins_nibbles(%08x)\n",
				threadIdx.x, VEC, tile->bins_nibbles[VEC / 8][LOAD]);
*/
			// Decode prefix bytes for first four keys
			tile->load_prefix_bytes[VEC / 4][LOAD] += util::PRMT(
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD],
				tile->bins_nibbles[VEC / 8][LOAD]);

			// Decode scan low and high packed words for first four keys
			int warpscan_prefix[2];
			warpscan_prefix[0] = util::PRMT(
				tile->warpscan_shorts[LOAD][0],
				tile->warpscan_shorts[LOAD][1],
				tile->bins_nibbles[VEC / 8][LOAD]);

			warpscan_prefix[1] = util::PRMT(
				tile->warpscan_shorts[LOAD][2],
				tile->warpscan_shorts[LOAD][3],
				tile->bins_nibbles[VEC / 8][LOAD]);

			// Low
			int packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x5140) +
				util::PRMT(								// Raking scan component (lower bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4140);

			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 0] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 1] = packed_scatter >> 16;

			// High
			packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x7362) +
				util::PRMT(								// Raking scan component (upper bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4342);

			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 2] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 3] = packed_scatter >> 16;

		}
	};


	/**
	 * ExtractRanks (VEC % 8 == 4)
	 */
	template <int LOAD, int VEC>
	struct ExtractRanks<LOAD, VEC, 4>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile)
		{
			int upper_bins_nibbles = tile->bins_nibbles[VEC / 8][LOAD] >> 16;

			// Decode prefix bytes for second four keys
			tile->load_prefix_bytes[VEC / 4][LOAD] += util::PRMT(
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD],
				upper_bins_nibbles);

			// Decode scan low and high packed words for second four keys
			int warpscan_prefix[2];
			warpscan_prefix[0] = util::PRMT(
				tile->warpscan_shorts[LOAD][0],
				tile->warpscan_shorts[LOAD][1],
				upper_bins_nibbles);

			warpscan_prefix[1] = util::PRMT(
				tile->warpscan_shorts[LOAD][2],
				tile->warpscan_shorts[LOAD][3],
				upper_bins_nibbles);

			// Low
			int packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x5140) +
				util::PRMT(								// Raking scan component (lower bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4140);

			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 0] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 1] = packed_scatter >> 16;

			// High
			packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x7362) +
				util::PRMT(								// Raking scan component (upper bytes from each half)
					tile->load_prefix_bytes[VEC / 4][LOAD],
					0,
					0x4342);

			packed_scatter <<= 2;

			tile->local_ranks[LOAD][VEC + 2] = packed_scatter & 0x0000ffff;
			tile->local_ranks[LOAD][VEC + 3] = packed_scatter >> 16;
		}
	};



	//---------------------------------------------------------------------
	// IterateTileElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int LOAD, int VEC, int dummy = 0>
	struct IterateTileElements
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			Dispatch *dispatch = (Dispatch *) tile;

			// Decode the bin for this key
			int bin = dispatch->DecodeBin(tile->keys[LOAD][VEC], cta);

			const int BITS_PER_NIBBLE = 4;
			int shift = bin * BITS_PER_NIBBLE;

			// Initialize exclusive scan bytes
			if (VEC == 0) {

				tile->load_prefix_bytes[VEC / 4][LOAD] = 0;

			} else {
				int prev_counts_nibbles = tile->counts_nibbles[0][LOAD] >> shift;

				if ((VEC & 3) == 0) {

					tile->load_prefix_bytes[VEC / 4][LOAD] = prev_counts_nibbles & 0xf;

				} else if ((VEC & 7) < 4) {

					util::BFI(
						tile->load_prefix_bytes[VEC / 4][LOAD],
						tile->load_prefix_bytes[VEC / 4][LOAD],
						prev_counts_nibbles,
						8 * (VEC & 7),
						BITS_PER_NIBBLE);

				} else {

					util::BFI(
						tile->load_prefix_bytes[VEC / 4][LOAD],
						tile->load_prefix_bytes[VEC / 4][LOAD],
						prev_counts_nibbles,
						8 * ((VEC & 7) - 4),
						BITS_PER_NIBBLE);
				}
			}

			// Initialize counts nibbles
			if (VEC == 0) {
				tile->counts_nibbles[0][LOAD] = 1 << shift;
			} else {
				util::SHL_ADD(
					tile->counts_nibbles[0][LOAD],
					1,
					shift,
					tile->counts_nibbles[0][LOAD]);
			}

			// Initialize bins nibbles
			if ((VEC & 7) == 0) {
				tile->bins_nibbles[VEC / 8][LOAD] = bin;

			} else {
				util::BFI(
					tile->bins_nibbles[VEC / 8][LOAD],
					tile->bins_nibbles[VEC / 8][LOAD],
					bin,
					4 * (VEC & 7),
					4);
			}

			// Next vector element
			IterateTileElements<LOAD, VEC + 1>::DecodeKeys(cta, tile);
		}


		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			if (VEC == 0) {

				const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

				// Extract prefix bytes from bytes raking grid
				tile->counts_bytes[0][LOAD] = cta->byte_grid_details.lane_partial[0][LANE_OFFSET];
				tile->counts_bytes[1][LOAD] = cta->byte_grid_details.lane_partial[1][LANE_OFFSET];

				// Extract warpscan shorts
				const int LOAD_RAKING_TID_OFFSET = (KernelPolicy::THREADS * LOAD) >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG;

				int base_raking_tid = threadIdx.x >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG;
				tile->warpscan_shorts[LOAD][0] = cta->smem_storage.short_prefixes[0][base_raking_tid + LOAD_RAKING_TID_OFFSET];
				tile->warpscan_shorts[LOAD][1] = cta->smem_storage.short_prefixes[1][base_raking_tid + LOAD_RAKING_TID_OFFSET];
				tile->warpscan_shorts[LOAD][2] = cta->smem_storage.short_prefixes[0][base_raking_tid + LOAD_RAKING_TID_OFFSET + (RAKING_THREADS / 2)];
				tile->warpscan_shorts[LOAD][3] = cta->smem_storage.short_prefixes[1][base_raking_tid + LOAD_RAKING_TID_OFFSET + (RAKING_THREADS / 2)];
			}

			ExtractRanks<LOAD, VEC>::Invoke(cta, tile);
/*
			printf("tid(%d) vec(%d) key(%d) scatter(%d)\n",
				threadIdx.x,
				VEC,
				tile->keys[LOAD][VEC],
				tile->local_ranks[LOAD][VEC]);
*/
			// Next vector element
			IterateTileElements<LOAD, VEC + 1>::ComputeRanks(cta, tile);
		}
	};



	/**
	 * IterateTileElements next load
	 */
	template <int LOAD, int dummy>
	struct IterateTileElements<LOAD, LOAD_VEC_SIZE, dummy>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			// Expand nibble-packed counts into pair of byte-packed counts
			util::NibblesToBytes(
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD],
				tile->counts_nibbles[0][LOAD]);

			const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

			// Place keys into raking grid
			cta->byte_grid_details.lane_partial[0][LANE_OFFSET] = tile->counts_bytes[0][LOAD];
			cta->byte_grid_details.lane_partial[1][LANE_OFFSET] = tile->counts_bytes[1][LOAD];
/*
			printf("Tid %u load %u:\t,"
				"load_prefix_bytes[0](%08x), "
				"load_prefix_bytes[1](%08x), "
				"bins_nibbles(%08x), "
				"counts_bytes[0](%08x), "
				"counts_bytes[1](%08x), "
				"\n",
				threadIdx.x, LOAD,
				tile->load_prefix_bytes[0][LOAD],
				tile->load_prefix_bytes[1][LOAD],
				tile->bins_nibbles[VEC / 8][LOAD],
				tile->counts_bytes[0][LOAD],
				tile->counts_bytes[1][LOAD]);
*/
			// First vector element, next load
			IterateTileElements<LOAD + 1, 0>::DecodeKeys(cta, tile);
		}

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// First vector element, next load
			IterateTileElements<LOAD + 1, 0>::ComputeRanks(cta, tile);
		}

	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateTileElements<LOADS_PER_TILE, 0, dummy>
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
	 * SOA scan operator (independent addition)
	 */
	struct SoaSumOp
	{
		enum {
			IDENTITY_STRIDES = true,			// There is an "identity" region of warpscan storage exists for strides to index into
		};

		// Tuple of partial-flag type
		typedef util::Tuple<int, int> TileTuple;

		// Scan operator
		__device__ __forceinline__ TileTuple operator()(
			const TileTuple &first,
			const TileTuple &second)
		{
			return TileTuple(first.t0 + second.t0, first.t1 + second.t1);
		}

		// Identity operator
		__device__ __forceinline__ TileTuple operator()()
		{
			return TileTuple(0,0);
		}

		template <typename WarpscanT>
		static __device__ __forceinline__ TileTuple WarpScanInclusive(
			TileTuple &total,
			TileTuple &input_partial,
			WarpscanT warpscan0,
			WarpscanT warpscan1)
		{
			// SOA type of warpscan storage
			typedef util::Tuple<WarpscanT, WarpscanT> WarpscanSoa;

			WarpscanSoa warpscan_soa(warpscan0, warpscan1);
			SoaSumOp scan_op;

			TileTuple current_partial;
			TileTuple inclusive_partial;

			// Extract input partial
			warpscan_soa.Get(input_partial, 1, threadIdx.x - 0);

			warpscan_soa.Get(current_partial, 1, threadIdx.x - 1);
			inclusive_partial = scan_op(current_partial, input_partial);
			warpscan_soa.Set(inclusive_partial, 1, threadIdx.x);

			warpscan_soa.Get(current_partial, 1, threadIdx.x - 2);
			inclusive_partial = scan_op(current_partial, inclusive_partial);
			warpscan_soa.Set(inclusive_partial, 1, threadIdx.x);

			warpscan_soa.Get(current_partial, 1, threadIdx.x - 4);
			inclusive_partial = scan_op(current_partial, inclusive_partial);
			warpscan_soa.Set(inclusive_partial, 1, threadIdx.x);

			warpscan_soa.Get(current_partial, 1, threadIdx.x - 8);
			inclusive_partial = scan_op(current_partial, inclusive_partial);
			warpscan_soa.Set(inclusive_partial, 1, threadIdx.x);

			warpscan_soa.Get(current_partial, 1, threadIdx.x - 16);
			inclusive_partial = scan_op(current_partial, inclusive_partial);
			warpscan_soa.Set(inclusive_partial, 1, threadIdx.x);

			warpscan_soa.Get(total, 1, 31);

			return inclusive_partial;
		}
	};


	/**
	 * Scan Tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScanTile(Cta *cta)
	{
		typedef typename SoaSumOp::TileTuple TileTuple;

		Dispatch *dispatch = (Dispatch*) this;

		// Decode bins and place keys into grid
		IterateTileElements<0, 0>::DecodeKeys(cta, dispatch);

		__syncthreads();

		// Use our raking threads to, in aggregate, scan the composite counter lanes
		if (threadIdx.x < RAKING_THREADS) {
/*
			if (threadIdx.x == 0) {
				printf("ByteGrid:\n");
				KernelPolicy::ByteGrid::Print();
				printf("\n");
			}
*/
			// Upsweep rake
			int partial_bytes = util::scan::SerialScan<KernelPolicy::ByteGrid::PARTIALS_PER_SEG>::Invoke(
				cta->byte_grid_details.raking_segment,
				0);

			// Trade
			int offset = (threadIdx.x < RAKING_THREADS / 2) ?
				0 :
				(2 * B40C_WARP_THREADS(CUDA_ARCH)) - 16;

			// Unpack byte-packed partial sum into short-packed partial sums
			cta->smem_storage.warpscan[0][1][threadIdx.x + offset] 			= util::PRMT(partial_bytes, 0, 0x4240);
			cta->smem_storage.warpscan[0][1][threadIdx.x + offset + 16] 	= util::PRMT(partial_bytes, 0, 0x4341);

			// Perform structure-of-arrays warpscan
			TileTuple total;
			TileTuple partial_shorts;
			TileTuple inclusive_partial = SoaSumOp::WarpScanInclusive(
				total,
				partial_shorts,
				cta->smem_storage.warpscan[0],
				cta->smem_storage.warpscan[1]);

/*
			printf("\t\tRaking thread %d reduced partial(%08x), extracted to ((%u,%u),(%u,%u))\n",
				threadIdx.x,
				partial_bytes,
				partial_shorts.t0 >> 16, partial_shorts.t0 & 0x0000ffff,
				partial_shorts.t1 >> 16, partial_shorts.t1 & 0x0000ffff);

			printf("Raking tid %d with inclusive_partial((%u,%u),(%u,%u)) and sums((%u,%u),(%u,%u))\n",
				threadIdx.x,
				inclusive_partial.t0 >> 16, inclusive_partial.t0 & 0x0000ffff,
				inclusive_partial.t1 >> 16, inclusive_partial.t1 & 0x0000ffff,
				total.t0 >> 16, total.t0 & 0x0000ffff,
				total.t1 >> 16, total.t1 & 0x0000ffff);
*/

			// Propagate the bottom total halves into the top inclusive partial halves
			inclusive_partial.t0 = util::SHL_ADD_C(total.t0, 16, inclusive_partial.t0);
			inclusive_partial.t1 = util::SHL_ADD_C(total.t1, 16, inclusive_partial.t1);

			// Sum of the lower inclusive partial in both halves
			int lower_addend = util::PRMT(total.t0, total.t0, 0x1032);
			lower_addend += total.t0;

			// Add it into the upper inclusive partial
			inclusive_partial.t1 += lower_addend;

			// Create exclusive partial
			TileTuple exclusive_partial(
				inclusive_partial.t0 - partial_shorts.t0,
				inclusive_partial.t1 - partial_shorts.t1);
/*
			printf("Raking tid %d with exclusive_partial((%u,%u),(%u,%u))\n",
				threadIdx.x,
				exclusive_partial.t0 >> 16, exclusive_partial.t0 & 0x0000ffff,
				exclusive_partial.t1 >> 16, exclusive_partial.t1 & 0x0000ffff);
*/
			// First half of raking threads hold bins (0,2),(4,6) in t0, t1
			// Second half of raking threads hold bins (1,3),(5,7) in t0, t1

			// Place short-packed partials back into smem for trading
			cta->smem_storage.warpscan[0][1][threadIdx.x] = exclusive_partial.t0;
			cta->smem_storage.warpscan[1][1][threadIdx.x] = exclusive_partial.t1;

			// Interleave:
			// First half of raking threads:	(0L, 1L, 2L, 3L),(4L, 5L, 6L, 7L)
			// Second half of raking threads: 	(0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H).
			int prmt = (threadIdx.x < (RAKING_THREADS / 2)) ? 0x6240 : 0x3715;
			int other = (threadIdx.x + (RAKING_THREADS / 2)) & (RAKING_THREADS - 1);
			int a = exclusive_partial.t0;													// 0,2
			int b = cta->smem_storage.warpscan[0][1][other];								// 1,3
			int c = exclusive_partial.t1;													// 4,6
			int d = cta->smem_storage.warpscan[1][1][other];								// 5,7

			cta->smem_storage.short_prefixes[0][threadIdx.x] =
				util::PRMT(a, b, prmt);
			cta->smem_storage.short_prefixes[1][threadIdx.x] =
				util::PRMT(c, d, prmt);


			// Update digit-carry
			SizeT my_carry;
			if (threadIdx.x < KernelPolicy::BINS) {
				// Add the previous tile's inclusive-scan to the running bin-carry
				my_carry = cta->smem_storage.bin_carry[threadIdx.x] +
					cta->smem_storage.bin_inclusive[1][threadIdx.x];
			}


			// Save off bin inclusive scans
			const int BIN_INCLUSIVE_MASK = (RAKING_THREADS / 2) - 1;

			if ((threadIdx.x & BIN_INCLUSIVE_MASK) == BIN_INCLUSIVE_MASK) {
				int base = threadIdx.x >> (LOG_RAKING_THREADS - 1);
				cta->smem_storage.bin_inclusive[1][base + 0] = inclusive_partial.t0 & 0x0000ffff;
				cta->smem_storage.bin_inclusive[1][base + 2] = inclusive_partial.t0 >> 16;;
				cta->smem_storage.bin_inclusive[1][base + 4] = inclusive_partial.t1 & 0x0000ffff;
				cta->smem_storage.bin_inclusive[1][base + 6] = inclusive_partial.t1 >> 16;;
			}

			// Update carry
			if (threadIdx.x < KernelPolicy::BINS) {

				// Subtract the bin prefix from the running carry (to offset threadIdx during scatter)
				int bin_exclusive = cta->smem_storage.bin_inclusive[1][threadIdx.x - 1];
				cta->smem_storage.bin_carry[threadIdx.x] = my_carry - bin_exclusive;

//				printf("bin (%d) has exclusive(%d)\n", threadIdx.x, bin_exclusive);
			}
		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateTileElements<0, 0>::ComputeRanks(cta, dispatch);
	}



	/**
	 * DecodeAndScatterKey
	 */
	template <int ELEMENT, typename Cta>
	__device__ __forceinline__ void DecodeAndScatterKey(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		Dispatch *dispatch = (Dispatch*) this;
		KeyType *linear_keys = (KeyType *) keys;

		int bin = dispatch->DecodeBin(linear_keys[ELEMENT], cta);
		SizeT carry_offset = cta->smem_storage.bin_carry[bin];

		int tile_element = threadIdx.x + (KernelPolicy::THREADS * ELEMENT);

		if ((guarded_elements >= KernelPolicy::TILE_ELEMENTS) || (tile_element < guarded_elements)) {

			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				linear_keys[ELEMENT],
				cta->d_out_keys + threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + carry_offset);
		}

/*
		printf("Tid %d scattering key[%d] (%d) with carry_bin %d to offset %d\n",
			threadIdx.x,
			ELEMENT,
			linear_keys[ELEMENT],
			cta->smem_storage.bin_carry[bin],
			threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + carry_offset);
*/
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
		// DecodeAndScatterKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeAndScatterKeys(
			Cta *cta,
			Tile *tile,
			const SizeT &guarded_elements)
		{
			tile->DecodeAndScatterKey<ELEMENT>(cta, guarded_elements);

			IterateElements<ELEMENT + 1>::DecodeAndScatterKeys(cta, tile, guarded_elements);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateElements<TILE_ELEMENTS_PER_THREAD, dummy>
	{
		// DecodeAndScatterKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeAndScatterKeys(
			Cta *cta, Tile *tile, const SizeT &guarded_elements) {}
	};



	//---------------------------------------------------------------------
	// Partition/scattering specializations
	//---------------------------------------------------------------------


	template <
		ScatterStrategy SCATTER_STRATEGY,
		int dummy = 0>
	struct PartitionTile;



	/**
	 * Specialized for two-phase scatter, keys-only
	 */
	template <
		ScatterStrategy SCATTER_STRATEGY,
		int dummy>
	struct PartitionTile
	{
		enum {
			MEM_BANKS 					= 1 << B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__),
			DIGITS_PER_SCATTER_PASS 	= KernelPolicy::WARPS * (B40C_WARP_THREADS(__B40C_CUDA_ARCH__) / (MEM_BANKS)),
			SCATTER_PASSES 				= KernelPolicy::BINS / DIGITS_PER_SCATTER_PASS,
		};

		template <typename T>
		static __device__ __forceinline__ void Nop(T &t) {}


		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(
			SizeT pack_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load keys
/*
			tile->LoadKeys(cta, cta_offset, guarded_elements);
*/

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

			// Scan tile
			tile->ScanTile(cta);

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

			// Gather keys linearly from smem (vec-1)
			util::io::LoadTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0,
				KernelPolicy::THREADS,
				util::io::ld::NONE,
				false>::LoadValid(									// No need to check alignment
					(KeyType (*)[1]) tile->keys,
					cta->smem_storage.key_exchange,
					0);

			__syncthreads();

			// Scatter keys to global bin partitions
			IterateElements<0>::DecodeAndScatterKeys(cta, tile, guarded_elements);

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

