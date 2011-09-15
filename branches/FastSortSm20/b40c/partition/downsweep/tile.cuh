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
		LOADS_PER_CYCLE 			= KernelPolicy::LOADS_PER_CYCLE,
		CYCLES_PER_TILE 			= KernelPolicy::CYCLES_PER_TILE,
		TILE_ELEMENTS_PER_THREAD 	= KernelPolicy::TILE_ELEMENTS_PER_THREAD,
		SCAN_LANES_PER_CYCLE		= KernelPolicy::SCAN_LANES_PER_CYCLE,

		LANE_ROWS_PER_LOAD 			= KernelPolicy::ByteGrid::ROWS_PER_LANE / KernelPolicy::LOADS_PER_CYCLE,
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


	// The keys (and values) this thread will read this cycle
	KeyType 	keys[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];

	// For each load:
	// 		counts_nibbles contains the bin counts within nibbles ordered right to left
	// 		bins_nibbles contains the bin for each key within nibbles ordered right to left
	// 		load_prefix_bytes contains the exclusive scan for each key within nibbles ordered right to left

	int 		bins_nibbles[CYCLES_PER_TILE][LOADS_PER_CYCLE];

	int 		counts_nibbles[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	int			counts_bytes0[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	int			counts_bytes1[CYCLES_PER_TILE][LOADS_PER_CYCLE];

	int 		load_prefix_bytes0[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	int 		load_prefix_bytes1[CYCLES_PER_TILE][LOADS_PER_CYCLE];

	int 		warpscan_shorts[CYCLES_PER_TILE][LOADS_PER_CYCLE][4];

	int 		local_ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];		// The local rank of each key
	SizeT 		scatter_offsets[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];	// The global rank of each key


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
	template <int CYCLE, int LOAD, int VEC>
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
	// Cycle Methods
	//---------------------------------------------------------------------

	/**
	 * ExtractRanks
	 */
	template <int CYCLE, int LOAD, int VEC>
	struct ExtractRanks
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile) {}
	};


	/**
	 * ExtractRanks (VEC == 0)
	 */
	template <int CYCLE, int LOAD>
	struct ExtractRanks<CYCLE, LOAD, 0>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile)
		{
			const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

			// Extract prefix bytes from bytes raking grid
			tile->counts_bytes0[CYCLE][LOAD] = cta->byte_grid_details.lane_partial[0][LANE_OFFSET];
			tile->counts_bytes1[CYCLE][LOAD] = cta->byte_grid_details.lane_partial[1][LANE_OFFSET];

			// Decode prefix bytes for first four keys
			tile->load_prefix_bytes0[CYCLE][LOAD] += util::PRMT(
				tile->counts_bytes0[CYCLE][LOAD],
				tile->counts_bytes1[CYCLE][LOAD],
				tile->bins_nibbles[CYCLE][LOAD]);

			// Extract warpscan shorts
			const int LOAD_RAKING_TID_OFFSET = 2 * ((KernelPolicy::THREADS * LOAD) >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG);

			int base_raking_tid = 2 * (threadIdx.x >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG);
			tile->warpscan_shorts[CYCLE][LOAD][0] = cta->smem_storage.short_prefixes_a[base_raking_tid + LOAD_RAKING_TID_OFFSET];
			tile->warpscan_shorts[CYCLE][LOAD][1] = cta->smem_storage.short_prefixes_a[base_raking_tid + LOAD_RAKING_TID_OFFSET + 1];
			tile->warpscan_shorts[CYCLE][LOAD][2] = cta->smem_storage.short_prefixes_a[base_raking_tid + LOAD_RAKING_TID_OFFSET + RAKING_THREADS];
			tile->warpscan_shorts[CYCLE][LOAD][3] = cta->smem_storage.short_prefixes_a[base_raking_tid + LOAD_RAKING_TID_OFFSET + RAKING_THREADS + 1];

			// Decode scan low and high packed words for first four keys
			int warpscan_prefix[2];
			warpscan_prefix[0] = util::PRMT(
				tile->warpscan_shorts[CYCLE][LOAD][0],
				tile->warpscan_shorts[CYCLE][LOAD][1],
				tile->bins_nibbles[CYCLE][LOAD]);

			warpscan_prefix[1] = util::PRMT(
				tile->warpscan_shorts[CYCLE][LOAD][2],
				tile->warpscan_shorts[CYCLE][LOAD][3],
				tile->bins_nibbles[CYCLE][LOAD]);

			// Low
			int packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x5140) +
				util::PRMT(								// Raking scan component (lower bytes from each half)
					tile->load_prefix_bytes0[CYCLE][LOAD],
					0,
					0x4140);

			tile->local_ranks[CYCLE][LOAD][0] = packed_scatter & 0x0000ffff;
			tile->local_ranks[CYCLE][LOAD][1] = packed_scatter >> 16;

			// High
			packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x7362) +
				util::PRMT(								// Raking scan component (upper bytes from each half)
					tile->load_prefix_bytes0[CYCLE][LOAD],
					0,
					0x4342);

			tile->local_ranks[CYCLE][LOAD][2] = packed_scatter & 0x0000ffff;
			tile->local_ranks[CYCLE][LOAD][3] = packed_scatter >> 16;

		}
	};


	/**
	 * ExtractRanks (VEC == 4)
	 */
	template <int CYCLE, int LOAD>
	struct ExtractRanks<CYCLE, LOAD, 4>
	{
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void Invoke(Cta *cta, Tile *tile)
		{
			int upper_bins_nibbles = tile->bins_nibbles[CYCLE][LOAD] >> 16;

			// Decode prefix bytes for second four keys
			tile->load_prefix_bytes1[CYCLE][LOAD] += util::PRMT(
				tile->counts_bytes0[CYCLE][LOAD],
				tile->counts_bytes1[CYCLE][LOAD],
				upper_bins_nibbles);

			// Decode scan low and high packed words for second four keys
			int warpscan_prefix[2];
			warpscan_prefix[0] = util::PRMT(
				tile->warpscan_shorts[CYCLE][LOAD][0],
				tile->warpscan_shorts[CYCLE][LOAD][1],
				upper_bins_nibbles);

			warpscan_prefix[1] = util::PRMT(
				tile->warpscan_shorts[CYCLE][LOAD][2],
				tile->warpscan_shorts[CYCLE][LOAD][3],
				upper_bins_nibbles);

			// Low
			int packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x5140) +
				util::PRMT(								// Raking scan component (lower bytes from each half)
					tile->load_prefix_bytes0[CYCLE][LOAD],
					0,
					0x4140);

			tile->local_ranks[CYCLE][LOAD][4] = packed_scatter & 0x0000ffff;
			tile->local_ranks[CYCLE][LOAD][5] = packed_scatter >> 16;

			// High
			packed_scatter =
				util::PRMT(								// Warpscan component (de-interleaved)
					warpscan_prefix[0],
					warpscan_prefix[1],
					0x7362) +
				util::PRMT(								// Raking scan component (upper bytes from each half)
					tile->load_prefix_bytes0[CYCLE][LOAD],
					0,
					0x4342);

			tile->local_ranks[CYCLE][LOAD][6] = packed_scatter & 0x0000ffff;
			tile->local_ranks[CYCLE][LOAD][7] = packed_scatter >> 16;
		}
	};



	//---------------------------------------------------------------------
	// IterateCycleElements Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next vector element
	 */
	template <int CYCLE, int LOAD, int VEC, int dummy = 0>
	struct IterateCycleElements
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			Dispatch *dispatch = (Dispatch *) tile;

			// Decode the bin for this key
			int bin = dispatch->DecodeBin(tile->keys[CYCLE][LOAD][VEC], cta);

			const int BITS_PER_NIBBLE = 4;
			int shift = bin * BITS_PER_NIBBLE;

			// Initialize exclusive scan bytes
			if (VEC == 0) {
				tile->load_prefix_bytes0[CYCLE][LOAD] = 0;

			} else if (VEC == 4) {
				tile->load_prefix_bytes1[CYCLE][LOAD] = 0;

			} else {
				int prev_counts_nibbles = tile->counts_nibbles[CYCLE][LOAD] >> shift;
				if (VEC < 4) {
					util::BFI(
						tile->load_prefix_bytes0[CYCLE][LOAD],
						tile->load_prefix_bytes0[CYCLE][LOAD],
						prev_counts_nibbles,
						8 * VEC,
						BITS_PER_NIBBLE);
				} else {
					util::BFI(
						tile->load_prefix_bytes1[CYCLE][LOAD],
						tile->load_prefix_bytes1[CYCLE][LOAD],
						prev_counts_nibbles,
						8 * (VEC - 4),
						BITS_PER_NIBBLE);
				}
			}

			// Initialize counts and bins nibbles
			if (VEC == 0) {
				tile->counts_nibbles[CYCLE][LOAD] = 1 << shift;
				tile->bins_nibbles[CYCLE][LOAD] = bin;

			} else {
				util::BFI(
					tile->bins_nibbles[CYCLE][LOAD],
					tile->bins_nibbles[CYCLE][LOAD],
					bin,
					4 * VEC,
					4);

				util::SHL_ADD(
					tile->counts_nibbles[CYCLE][LOAD],
					1,
					shift,
					tile->counts_nibbles[CYCLE][LOAD]);
			}

			// Next vector element
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::DecodeKeys(cta, tile);
		}


		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			ExtractRanks<CYCLE, LOAD, VEC>::Invoke(cta, tile);

/*
			printf("tid(%d) vec(%d) key(%d) scatter(%d)\n",
				threadIdx.x,
				VEC,
				tile->keys[CYCLE][LOAD][VEC],
				tile->local_ranks[CYCLE][LOAD][VEC]);
*/

			// Next vector element
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::ComputeRanks(cta, tile);
		}
	};



	/**
	 * IterateCycleElements next load
	 */
	template <int CYCLE, int LOAD, int dummy>
	struct IterateCycleElements<CYCLE, LOAD, LOAD_VEC_SIZE, dummy>
	{
		// DecodeKeys
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeKeys(Cta *cta, Tile *tile)
		{
			// Expand nibble-packed counts into pair of byte-packed counts
			util::NibblesToBytes(
				tile->counts_bytes0[CYCLE][LOAD],
				tile->counts_bytes1[CYCLE][LOAD],
				tile->counts_nibbles[CYCLE][LOAD]);

			const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

			// Place keys into raking grid
			cta->byte_grid_details.lane_partial[0][LANE_OFFSET] = tile->counts_bytes0[CYCLE][LOAD];
			cta->byte_grid_details.lane_partial[1][LANE_OFFSET] = tile->counts_bytes1[CYCLE][LOAD];
/*
			printf("Tid %u cycle %u load %u:\t,"
				"load_prefix_bytes0(%08x), "
				"load_prefix_bytes1(%08x), "
				"bins_nibbles(%08x), "
				"counts_bytes0(%08x), "
				"counts_bytes1(%08x), "
				"\n",
				threadIdx.x, CYCLE, LOAD,
				tile->load_prefix_bytes0[CYCLE][LOAD],
				tile->load_prefix_bytes1[CYCLE][LOAD],
				tile->bins_nibbles[CYCLE][LOAD],
				tile->counts_bytes0[CYCLE][LOAD],
				tile->counts_bytes1[CYCLE][LOAD]);
*/

			// First vector element, next load
			IterateCycleElements<CYCLE, LOAD + 1, 0>::DecodeKeys(cta, tile);
		}

		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeRanks(Cta *cta, Tile *tile)
		{
			// First vector element, next load
			IterateCycleElements<CYCLE, LOAD + 1, 0>::ComputeRanks(cta, tile);
		}

	};

	/**
	 * Terminate iteration
	 */
	template <int CYCLE, int dummy>
	struct IterateCycleElements<CYCLE, LOADS_PER_CYCLE, 0, dummy>
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
			TileTuple partial,
			WarpscanT warpscan_low,
			WarpscanT warpscan_high)
		{
			// SOA type of warpscan storage
			typedef util::Tuple<WarpscanT, WarpscanT> WarpscanSoa;

			WarpscanSoa warpscan_soa(warpscan_low, warpscan_high);
			SoaSumOp scan_op;

			// Exclusive warp scan, get total
			TileTuple inclusive_partial = util::scan::soa::WarpSoaScan<
				LOG_WARPSCAN_THREADS,
				false>::Scan(
					partial,
					total,
					warpscan_soa,
					scan_op);

			return inclusive_partial;
		}
	};


	/**
	 * Scan Cycle
	 */
	template <int CYCLE, typename Cta>
	__device__ __forceinline__ void ScanCycle(Cta *cta)
	{
		typedef typename SoaSumOp::TileTuple TileTuple;

		Dispatch *dispatch = (Dispatch*) this;

		// Decode bins and place keys into grid
		IterateCycleElements<CYCLE, 0, 0>::DecodeKeys(cta, dispatch);

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

			// Unpack byte-packed partial sum into short-packed partial sums
			TileTuple partial_shorts(
				util::PRMT(partial_bytes, 0, 0x4240),
				util::PRMT(partial_bytes, 0, 0x4341));
/*
			printf("\t\tRaking thread %d reduced partial(%08x), extracted to ((%u,%u),(%u,%u))\n",
				threadIdx.x,
				partial_bytes,
				partial_shorts.t0 >> 16, partial_shorts.t0 & 0x0000ffff,
				partial_shorts.t1 >> 16, partial_shorts.t1 & 0x0000ffff);
*/
			// Perform structure-of-arrays warpscan

			TileTuple total;
			TileTuple inclusive_partial = SoaSumOp::WarpScanInclusive(
				total,
				partial_shorts,
				cta->smem_storage.warpscan_low,
				cta->smem_storage.warpscan_high);
/*
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


			// Take the bottom half of the lower inclusive partial
			// and add it into the top half (top half now contains sum of both halves of total.t0)
			int lower_addend = util::SHL_ADD_C(total.t0, 16, total.t0);

			// Duplicate the top half
			lower_addend = util::PRMT(lower_addend, 0, 0x3232);

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
			cta->smem_storage.warpscan_low[1][threadIdx.x] = exclusive_partial.t0;
			cta->smem_storage.warpscan_high[1][threadIdx.x] = exclusive_partial.t1;

			if (threadIdx.x < (RAKING_THREADS / 2)) {
				int a = exclusive_partial.t0;													// 0,2
				int b = cta->smem_storage.warpscan_low[1][threadIdx.x + (RAKING_THREADS / 2)];	// 1,3
				int c = exclusive_partial.t1;													// 4,6
				int d = cta->smem_storage.warpscan_high[1][threadIdx.x + (RAKING_THREADS / 2)];	// 5,7

				// (0L, 1L, 2L, 3L), (4L, 5L, 6L, 7L),
				// (0H, 1H, 2H, 3H), (4H, 5H, 6H, 7H).
				cta->smem_storage.short_prefixes_a[(threadIdx.x * 2)] =
					util::PRMT(a, b, 0x6240);
				cta->smem_storage.short_prefixes_a[(threadIdx.x * 2) + 1] =
					util::PRMT(c, d, 0x6240);
				cta->smem_storage.short_prefixes_a[(threadIdx.x * 2) + RAKING_THREADS] =
					util::PRMT(a, b, 0x7351);
				cta->smem_storage.short_prefixes_a[(threadIdx.x * 2) + RAKING_THREADS + 1] =
					util::PRMT(c, d, 0x7351);
			}

		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateCycleElements<CYCLE, 0, 0>::ComputeRanks(cta, dispatch);
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

			*(cta->d_out_keys + threadIdx.x + (KernelPolicy::THREADS * ELEMENT) + carry_offset) = linear_keys[ELEMENT];
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
	// IterateCycles Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next cycle
	 */
	template <int CYCLE, int dummy = 0>
	struct IterateCycles
	{
		// ScanCycles
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScanCycles(Cta *cta, Tile *tile)
		{
			tile->ScanCycle<CYCLE>(cta);
			IterateCycles<CYCLE + 1>::ScanCycles(cta, tile);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateCycles<CYCLES_PER_TILE, dummy>
	{
		// ScanCycles
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScanCycles(Cta *cta, Tile *tile) {}
	};


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
			SizeT cta_offset,
			const SizeT &guarded_elements,
			Cta *cta,
			Tile *tile)
		{
			// Load keys
			tile->LoadKeys(cta, cta_offset, guarded_elements);

			// Scan cycles
			IterateCycles<0>::ScanCycles(cta, tile);

			__syncthreads();

			// Scatter keys to smem by local rank
			util::io::ScatterTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0,
				KernelPolicy::THREADS,
				util::io::st::NONE>::Scatter(
					cta->smem_storage.key_exchange,
					(KeyType (*)[1]) tile->keys,
					(int (*)[1]) tile->local_ranks);

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
		SizeT cta_offset,
		const SizeT &guarded_elements,
		Cta *cta)
	{
		PartitionTile<KernelPolicy::SCATTER_STRATEGY>::Invoke(
			cta_offset,
			guarded_elements,
			cta,
			(Dispatch *) this);

	}

};


} // namespace downsweep
} // namespace partition
} // namespace b40c

