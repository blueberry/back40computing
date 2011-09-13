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
	// 		escan_bytes contains the exclusive scan for each key within nibbles ordered right to left

	int 		counts_nibbles[CYCLES_PER_TILE][LOADS_PER_CYCLE][1];
	int 		escan_bytes[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	int 		bins_nibbles[CYCLES_PER_TILE][LOADS_PER_CYCLE];

	int			counts_bytes[CYCLES_PER_TILE][LOADS_PER_CYCLE][1][2];

	int 		short_offsets[SCAN_LANES_PER_CYCLE][2];

	int 		local_ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];		// The local rank of each key
	SizeT 		scatter_offsets[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];	// The global rank of each key

	// Counts of my bin in each load in each cycle, valid in threads [0,BINS)
	int 		bin_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE];


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



	static __device__ __forceinline__ int MapBin(int native_bin)
	{
		// Remap bins
		// e.g., (0s, 1s, 2s, 3s, 4s, 5s, 6s, 7s) -> (0s, 4s, 1s, 5s, 2s, 6s, 3s, 7s)
		const int LUT0 = 0x05010400;
		const int LUT1 = 0x07030602;

		return util::PRMT(LUT0, LUT1, native_bin) & 0xff;
	}

	static __device__ __forceinline__ int UnmapBin(int mapped_bin)
	{
		// Unmap bins
		// e.g., (0, 1, 2, 3, 4, 5, 6, 7) -> (0, 2, 4, 6, 1, 3, 5, 7)
		const int LUT0 = 0x06040200;
		const int LUT1 = 0x07050301;

		return util::PRMT(LUT0, LUT1, mapped_bin) & 0xff;
	}


	/**
	 * DecodeKeys
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void DecodeKeys(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;

		// Update composite-counter
		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			// Decode the bin for this key
			int bin = dispatch->DecodeBin(keys[CYCLE][LOAD][VEC], cta);

			const int LOG_BITS_PER_NIBBLE = 2;
			const int BITS_PER_NIBBLE = 1 << LOG_BITS_PER_NIBBLE;

			bin = MapBin(bin);

			int shift = bin << LOG_BITS_PER_NIBBLE;

			if (VEC == 0) {

				counts_nibbles[CYCLE][LOAD][0] = 1 << shift;
				escan_bytes[CYCLE][LOAD] = 0;
				bins_nibbles[CYCLE][LOAD] = bin;

			} else {

				int prev_counts_nibbles = counts_nibbles[CYCLE][LOAD][0] >> shift;

				util::BFI(
					escan_bytes[CYCLE][LOAD],
					escan_bytes[CYCLE][LOAD],
					prev_counts_nibbles,
					8 * VEC,
					BITS_PER_NIBBLE);

				util::BFI(
					bins_nibbles[CYCLE][LOAD],
					bins_nibbles[CYCLE][LOAD],
					bin,
					4 * VEC,
					4);

				util::SHL_ADD(
					counts_nibbles[CYCLE][LOAD][0],
					1,
					shift,
					counts_nibbles[CYCLE][LOAD][0]);
			}

		} else {

			// Mooch
//			key_bins[CYCLE][LOAD][VEC] = INVALID_BIN;
		}
	}


	/**
	 * ExtractRanks
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void ExtractRanks(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;
		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			if (VEC == 0) {

				const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

				// Todo fix for other radix digits

				// Lane 0
				counts_bytes[CYCLE][LOAD][0][0] = cta->byte_grid_details.lane_partial[0][LANE_OFFSET];

				// Lane 1
				counts_bytes[CYCLE][LOAD][0][1] = cta->byte_grid_details.lane_partial[1][LANE_OFFSET];

				escan_bytes[CYCLE][LOAD] += util::PRMT(
					counts_bytes[CYCLE][LOAD][0][0],
					counts_bytes[CYCLE][LOAD][0][1],
					bins_nibbles[CYCLE][LOAD]);
/*
				printf("Downsweep thread %u cycle %u load %u:\t,"
					"escan_bytes(%x), "
					"bins_nibbles(%x), "
					"counts_bytes1(%08x), "
					"counts_bytes0(%08x), "
					"\n",
					threadIdx.x, CYCLE, LOAD,
					escan_bytes[CYCLE][LOAD],
					bins_nibbles[CYCLE][LOAD],
					counts_bytes[CYCLE][LOAD][0][1],
					counts_bytes[CYCLE][LOAD][0][0]);
*/
			}

			int nibble_prefix = util::BFE(escan_bytes[CYCLE][LOAD], VEC * 8, 8);

			int base_byte_raking_tid =
				(threadIdx.x + (KernelPolicy::THREADS * LOAD)) >> KernelPolicy::ByteGrid::LOG_PARTIALS_PER_SEG;

			int bin = util::BFE(bins_nibbles[CYCLE][LOAD], VEC * 4, 4);

			int lane = bin >> 2;
			int word = (bin >> 1) & 1;
			int select = bin & 1;

			int raking_tid = base_byte_raking_tid + (lane << KernelPolicy::ByteGrid::LOG_RAKING_THREADS_PER_LANE);
			short * short_offsets = (short *) &cta->smem_storage.short_deposits[word][raking_tid];
			int scan_prefix = short_offsets[select];

			local_ranks[CYCLE][LOAD][VEC] = scan_prefix + nibble_prefix;

/*
			bin = UnmapBin(bin);
			printf("\tExtract "
				"thread %u load %u vec %u\t:"
				"key(%u), "
				"bin(%u), "
				"base_byte_raking_tid(%u), "
				"lane(%u), "
				"word(%u), "
				"select(%u), "
				"nibble_prefix(%u),"
				"scan_prefix(%u), "
				"local_ranks(%u)\n",

				threadIdx.x, LOAD, VEC,
				keys[CYCLE][LOAD][VEC],
				bin,
				base_byte_raking_tid,
				lane,
				word,
				select,
				nibble_prefix,
				scan_prefix,
				local_ranks[CYCLE][LOAD][VEC]);
*/
		} else {
			// Put invalid keys just after the end of the valid swap exchange.
			local_ranks[CYCLE][LOAD][VEC] = KernelPolicy::TILE_ELEMENTS;
		}

	}


	/**
	 * UpdateRanks
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void UpdateRanks(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;
		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			// Update this key's rank with the bin-prefix for it's bin

			int bin = util::BFE(bins_nibbles[CYCLE][LOAD], VEC * 4, 4);

			local_ranks[CYCLE][LOAD][VEC] +=
				cta->smem_storage.bin_prefixes[CYCLE][LOAD][bin];
		}
	}


	/**
	 * UpdateGlobalOffsets
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta)
	{

		Dispatch *dispatch = (Dispatch *) this;
		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			int bin = util::BFE(bins_nibbles[CYCLE][LOAD], VEC * 4, 4);

			// Update this key's global scatter offset with its
			// cycle rank and with the bin-prefix for it's bin
			scatter_offsets[CYCLE][LOAD][VEC] =
				local_ranks[CYCLE][LOAD][VEC] +
				cta->smem_storage.bin_prefixes[CYCLE][LOAD][bin];
		}
	}



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
			tile->DecodeKeys<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::DecodeKeys(cta, tile);
		}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ExtractRanks(Cta *cta, Tile *tile)
		{
			tile->ExtractRanks<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::ExtractRanks(cta, tile);
		}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile)
		{
			tile->UpdateRanks<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::UpdateRanks(cta, tile);
		}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile)
		{
			tile->UpdateGlobalOffsets<CYCLE, LOAD, VEC>(cta);
			IterateCycleElements<CYCLE, LOAD, VEC + 1>::UpdateGlobalOffsets(cta, tile);
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
			util::NibblesToBytes(
				tile->counts_bytes[CYCLE][LOAD][0],
				tile->counts_nibbles[CYCLE][LOAD][0]);

			const int LANE_OFFSET = LOAD * LANE_STRIDE_PER_LOAD;

			// Todo fix for other radix digits
			cta->byte_grid_details.lane_partial[0][LANE_OFFSET] = tile->counts_bytes[CYCLE][LOAD][0][0];
			cta->byte_grid_details.lane_partial[1][LANE_OFFSET] = tile->counts_bytes[CYCLE][LOAD][0][1];
/*
			printf("Thread %u cycle %u load %u:\t,"
				"escan_bytes(%x), "
				"bins_nibbles(%x), "
				"counts_bytes1(%08x), "
				"counts_bytes0(%08x), "
				"lane_offset(%d)"
				"\n",
				threadIdx.x, CYCLE, LOAD,
				tile->escan_bytes[CYCLE][LOAD],
				tile->bins_nibbles[CYCLE][LOAD],
				tile->counts_bytes[CYCLE][LOAD][0][1],
				tile->counts_bytes[CYCLE][LOAD][0][0],
				LANE_OFFSET);
*/
			IterateCycleElements<CYCLE, LOAD + 1, 0>::DecodeKeys(cta, tile);
		}

		// ExtractRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ExtractRanks(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::ExtractRanks(cta, tile);
		}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::UpdateRanks(cta, tile);
		}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, LOAD + 1, 0>::UpdateGlobalOffsets(cta, tile);
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
		static __device__ __forceinline__ void ExtractRanks(Cta *cta, Tile *tile) {}

		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile) {}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile) {}
	};



	//---------------------------------------------------------------------
	// Tile Internal Methods
	//---------------------------------------------------------------------

	/**
	 * Scan Cycle
	 */
	template <int CYCLE, typename Cta>
	__device__ __forceinline__ void ScanCycle(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Decode bins and update 8-bit composite counters for the keys in this cycle
		IterateCycleElements<CYCLE, 0, 0>::DecodeKeys(cta, dispatch);

		__syncthreads();

		// Use our raking threads to, in aggregate, scan the composite counter lanes
		if (threadIdx.x < KernelPolicy::ByteGrid::RAKING_THREADS) {
/*
			if (threadIdx.x == 0) {
				printf("ByteGrid:\n");
				KernelPolicy::ByteGrid::Print();
				printf("\n");
				printf("ShortGrid:\n");
				KernelPolicy::ShortGrid::Print();
				printf("\n");
			}
*/
			// Upsweep rake
			int partial = util::scan::SerialScan<KernelPolicy::ByteGrid::PARTIALS_PER_SEG>::Invoke(
				cta->byte_grid_details.raking_segment,
				0);

			int halves[2];

			util::PRMT(halves[0], partial, 0, 0x4240);		// (d, c, b, a) -> (-, c, -, a),
			util::PRMT(halves[1], partial, 0, 0x4341);		// (d, c, b, a) -> (-, d, -, b),

			// raking tid < (RAKING_THREADS / 2) does even bins
			// raking tid >= (RAKING_THREADS / 2) does odd bins

			cta->short_grid_details.lane_partial[0][0] = halves[0];
			cta->short_grid_details.lane_partial[1][0] = halves[1];

			util::Sum<int> scan_op;
			util::scan::CooperativeGridScan<typename Cta::ShortGridDetails>::ScanTile(
				cta->short_grid_details,
				scan_op);

			int total = cta->short_grid_details.CumulativePartial();
			int addend = total << 16;
			halves[0] = cta->short_grid_details.lane_partial[0][0] + addend;
			halves[1] = cta->short_grid_details.lane_partial[1][0] + addend;

			// rebundle halves
			int rebundled_halves[2];
			rebundled_halves[0] = util::PRMT(halves[0], halves[1], 0x5410);
			rebundled_halves[1] = util::PRMT(halves[0], halves[1], 0x7632);

			// Store using short raking lanes
			cta->smem_storage.short_deposits[0][threadIdx.x] = rebundled_halves[0];
			cta->smem_storage.short_deposits[1][threadIdx.x] = rebundled_halves[1];
/*
			printf("\tRaking thread %d computed half0(%u, %u)\n",
				threadIdx.x,
				halves[0] & 0x0000ffff,
				(halves[0] >> 16) & 0x0000ffff);
			printf("\tRaking thread %d computed half1(%u, %u)\n",
				threadIdx.x,
				halves[1] & 0x0000ffff,
				(halves[1] >> 16) & 0x0000ffff);
*/
		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateCycleElements<CYCLE, 0, 0>::ExtractRanks(cta, dispatch);
	}


	/**
	 * RecoverBinCounts
	 *
	 * Called by threads [0, KernelPolicy::BINS)
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void RecoverBinCounts(
		int my_base_lane, int my_quad_byte, Cta *cta)
	{
		bin_counts[CYCLE][LOAD] =
			cta->smem_storage.lane_totals_c[CYCLE][LOAD][my_base_lane][0][my_quad_byte] +
			cta->smem_storage.lane_totals_c[CYCLE][LOAD][my_base_lane][1][my_quad_byte];
	}


	/**
	 * UpdateBinPrefixes
	 *
	 * Called by threads [0, KernelPolicy::BINS)
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void UpdateBinPrefixes(int bin_prefix, Cta *cta)
	{
		cta->smem_storage.bin_prefixes[CYCLE][LOAD][threadIdx.x] =
			bin_counts[CYCLE][LOAD] + bin_prefix;
	}


	/**
	 * DecodeGlobalOffsets
	 */
	template <int ELEMENT, typename Cta>
	__device__ __forceinline__ void DecodeGlobalOffsets(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		KeyType *linear_keys 	= (KeyType *) keys;
		SizeT *linear_offsets 	= (SizeT *) scatter_offsets;

		int bin = dispatch->DecodeBin(linear_keys[ELEMENT], cta);

		linear_offsets[ELEMENT] =
//			cta->smem_storage.bin_carry[bin] +							// mooch
			(KernelPolicy::THREADS * ELEMENT) + threadIdx.x;
/*
		printf("Tid %d scattering key[%d](%d) to offset %d\n",
			threadIdx.x,
			ELEMENT,
			linear_keys[ELEMENT],
			linear_offsets[ELEMENT]);
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
		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, 0, 0>::UpdateRanks(cta, tile);
			IterateCycles<CYCLE + 1>::UpdateRanks(cta, tile);
		}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile)
		{
			IterateCycleElements<CYCLE, 0, 0>::UpdateGlobalOffsets(cta, tile);
			IterateCycles<CYCLE + 1>::UpdateGlobalOffsets(cta, tile);
		}

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
		// UpdateRanks
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateRanks(Cta *cta, Tile *tile) {}

		// UpdateGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateGlobalOffsets(Cta *cta, Tile *tile) {}

		// ScanCycles
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ScanCycles(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// IterateCycleLoads Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next load
	 */
	template <int CYCLE, int LOAD, int dummy = 0>
	struct IterateCycleLoads
	{
		// RecoverBinCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverBinCounts(
			int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile)
		{
			tile->template RecoverBinCounts<CYCLE, LOAD>(my_base_lane, my_quad_byte, cta);
			IterateCycleLoads<CYCLE, LOAD + 1>::RecoverBinCounts(my_base_lane, my_quad_byte, cta, tile);
		}

		// UpdateBinPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateBinPrefixes(
			int bin_prefix, Cta *cta, Tile *tile)
		{
			tile->template UpdateBinPrefixes<CYCLE, LOAD>(bin_prefix, cta);
			IterateCycleLoads<CYCLE, LOAD + 1>::UpdateBinPrefixes(bin_prefix, cta, tile);
		}
	};


	/**
	 * Iterate next cycle
	 */
	template <int CYCLE, int dummy>
	struct IterateCycleLoads<CYCLE, LOADS_PER_CYCLE, dummy>
	{
		// RecoverBinCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverBinCounts(
			int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile)
		{
			IterateCycleLoads<CYCLE + 1, 0>::RecoverBinCounts(my_base_lane, my_quad_byte, cta, tile);
		}

		// UpdateBinPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateBinPrefixes(
			int bin_prefix, Cta *cta, Tile *tile)
		{
			IterateCycleLoads<CYCLE + 1, 0>::UpdateBinPrefixes(bin_prefix, cta, tile);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateCycleLoads<CYCLES_PER_TILE, 0, dummy>
	{
		// RecoverBinCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverBinCounts(int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile) {}

		// UpdateBinPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateBinPrefixes(int bin_prefix, Cta *cta, Tile *tile) {}
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
		// DecodeGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeGlobalOffsets(Cta *cta, Tile *tile)
		{
			tile->DecodeGlobalOffsets<ELEMENT>(cta);
			IterateElements<ELEMENT + 1>::DecodeGlobalOffsets(cta, tile);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateElements<TILE_ELEMENTS_PER_THREAD, dummy>
	{
		// DecodeGlobalOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void DecodeGlobalOffsets(Cta *cta, Tile *tile) {}
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

		/**
		 * Warp based scattering that does not cross alignment boundaries, e.g., for SM1.0-1.1
		 * coalescing rules
		 * /
		template <int PASS, int SCATTER_PASSES>
		struct WarpScatter
		{
			template <typename T, void Transform(T&), typename Cta>
			static __device__ __forceinline__ void ScatterPass(
				Cta *cta,
				T *exchange,
				T *d_out,
				const SizeT &valid_elements)
			{
				const int LOG_STORE_TXN_THREADS = B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__);
				const int STORE_TXN_THREADS = 1 << LOG_STORE_TXN_THREADS;

				int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
				int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;

				int my_digit = (PASS * DIGITS_PER_SCATTER_PASS) + store_txn_digit;

				if (my_digit < KernelPolicy::BINS) {

					int my_exclusive_scan = cta->smem_storage.bin_warpscan[1][my_digit - 1];
					int my_inclusive_scan = cta->smem_storage.bin_warpscan[1][my_digit];
					int my_digit_count = my_inclusive_scan - my_exclusive_scan;

					int my_carry = cta->smem_storage.bin_carry[my_digit] + my_exclusive_scan;
					int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));

					while (my_aligned_offset < my_digit_count) {

						if ((my_aligned_offset >= 0) && (my_exclusive_scan + my_aligned_offset < valid_elements)) {

							T datum = exchange[my_exclusive_scan + my_aligned_offset];
							Transform(datum);
							d_out[my_carry + my_aligned_offset] = datum;
						}
						my_aligned_offset += STORE_TXN_THREADS;
					}
				}

				WarpScatter<PASS + 1, SCATTER_PASSES>::template ScatterPass<T, Transform>(
					cta,
					exchange,
					d_out,
					valid_elements);
			}
		};

		// Terminate
		template <int SCATTER_PASSES>
		struct WarpScatter<SCATTER_PASSES, SCATTER_PASSES>
		{
			template <typename T, void Transform(T&), typename Cta>
			static __device__ __forceinline__ void ScatterPass(
				Cta *cta,
				T *exchange,
				T *d_out,
				const SizeT &valid_elements) {}
		};

		template <bool KEYS_ONLY, int dummy2 = 0>
		struct ScatterValues
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
				SizeT cta_offset,
				const SizeT &guarded_elements,
				const SizeT &valid_elements,
				Cta *cta,
				Tile *tile)
			{
				// Load values
				tile->LoadValues(cta, cta_offset, guarded_elements);

				// Scatter values to smem by local rank
				util::io::ScatterTile<
					KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
					0,
					KernelPolicy::THREADS,
					util::io::st::NONE>::Scatter(
						cta->smem_storage.value_exchange,
						(ValueType (*)[1]) tile->values,
						(int (*)[1]) tile->local_ranks);

				__syncthreads();

				if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

					WarpScatter<0, SCATTER_PASSES>::template ScatterPass<ValueType, Nop<ValueType> >(
						cta,
						cta->smem_storage.value_exchange,
						cta->d_out_values,
						valid_elements);

					__syncthreads();

				} else {

					// Gather values linearly from smem (vec-1)
					util::io::LoadTile<
						KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
						0,
						KernelPolicy::THREADS,
						util::io::ld::NONE,
						false>::LoadValid(									// No need to check alignment
							(ValueType (*)[1]) tile->values,
							cta->smem_storage.value_exchange,
							0);

					__syncthreads();

					// Scatter values to global bin partitions
					tile->ScatterValues(cta, valid_elements);
				}
			}
		};

		template <int dummy2>
		struct ScatterValues<true, dummy2>
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
					SizeT cta_offset,
					const SizeT &guarded_elements,
					const SizeT &valid_elements,
					Cta *cta,
					Tile *tile) {}
		};
*/

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



/*
			// Scan across bins
			if (threadIdx.x < KernelPolicy::BINS) {

				// Recover bin-counts from lane totals
				int my_base_lane = threadIdx.x >> 2;
				int my_quad_byte = threadIdx.x & 3;
				IterateCycleLoads<0, 0>::RecoverBinCounts(
					my_base_lane, my_quad_byte, cta, tile);

				// Scan across my bin counts for each load
				int tile_bin_total = util::scan::SerialScan<KernelPolicy::LOADS_PER_TILE>::Invoke(
					(int *) tile->bin_counts, 0);

				// Add the previous tile's inclusive-scan to the running bin-carry
				SizeT my_carry = cta->smem_storage.bin_carry[threadIdx.x] +
					cta->smem_storage.bin_warpscan[1][threadIdx.x];

				// Perform overflow-free inclusive SIMD Kogge-Stone across bins
				int tile_bin_inclusive = util::scan::WarpScan<KernelPolicy::LOG_BINS, false>::Invoke(
					tile_bin_total,
					cta->smem_storage.bin_warpscan);

				// Save inclusive scan in bin_warpscan
				cta->smem_storage.bin_warpscan[1][threadIdx.x] = tile_bin_inclusive;

				// Calculate exclusive scan
				int tile_bin_exclusive = tile_bin_inclusive - tile_bin_total;

				// Subtract the bin prefix from the running carry (to offset threadIdx during scatter)
				cta->smem_storage.bin_carry[threadIdx.x] = my_carry - tile_bin_exclusive;

				// Compute the bin prefixes for this tile for each load
				IterateCycleLoads<0, 0>::UpdateBinPrefixes(tile_bin_exclusive, cta, tile);
			}

			__syncthreads();

			// Update the local ranks in each load with the bin prefixes for the tile
			IterateCycles<0>::UpdateRanks(cta, tile);
*/

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

			SizeT valid_elements = tile->ValidElements(cta, guarded_elements);

/*
			if (SCATTER_STRATEGY == SCATTER_WARP_TWO_PHASE) {

				WarpScatter<0, SCATTER_PASSES>::template ScatterPass<KeyType, KernelPolicy::PostprocessKey>(
					cta,
					cta->smem_storage.key_exchange,
					cta->d_out_keys,
					valid_elements);

				__syncthreads();

			} else {
*/

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

				// Compute global scatter offsets for gathered keys
				IterateElements<0>::DecodeGlobalOffsets(cta, tile);

				// Scatter keys to global bin partitions
				tile->ScatterKeys(cta, valid_elements);

/*
			}

			// Partition values
			ScatterValues<KernelPolicy::KEYS_ONLY>::Invoke(
				cta_offset, guarded_elements, valid_elements, cta, tile);
*/
		}
	};


	/**
	 * Specialized for direct scatter
	 * /
	template <int dummy>
	struct PartitionTile<SCATTER_DIRECT, dummy>
	{
		template <bool KEYS_ONLY, int dummy2 = 0>
		struct ScatterValues
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
				SizeT cta_offset,
				const SizeT &guarded_elements,
				const SizeT &valid_elements,
				Cta *cta,
				Tile *tile)
			{
				// Load values
				tile->LoadValues(cta, cta_offset, guarded_elements);

				// Scatter values to global bin partitions
				tile->ScatterValues(cta, valid_elements);
			}
		};

		template <int dummy2>
		struct ScatterValues<true, dummy2>
		{
			template <typename Cta, typename Tile>
			static __device__ __forceinline__ void Invoke(
					SizeT cta_offset,
					const SizeT &guarded_elements,
					const SizeT &valid_elements,
					Cta *cta,
					Tile *tile) {}
		};

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

			// Scan across bins
			if (threadIdx.x < KernelPolicy::BINS) {

				// Recover bin-counts from lane totals
				int my_base_lane = threadIdx.x >> 2;
				int my_quad_byte = threadIdx.x & 3;
				IterateCycleLoads<0, 0>::RecoverBinCounts(
					my_base_lane, my_quad_byte, cta, tile);

				// Scan across my bin counts for each load
				int tile_bin_total = util::scan::SerialScan<KernelPolicy::LOADS_PER_TILE>::Invoke(
					(int *) tile->bin_counts, 0);

				// Add the previous tile's inclusive-scan to the running bin-carry
				SizeT my_carry = cta->smem_storage.bin_carry[threadIdx.x];

				// Update bin prefixes with the incoming carry
				IterateCycleLoads<0, 0>::UpdateBinPrefixes(my_carry, cta, tile);

				// Update carry
				cta->smem_storage.bin_carry[threadIdx.x] = my_carry + tile_bin_total;
			}

			__syncthreads();

			SizeT valid_elements = tile->ValidElements(cta, guarded_elements);

			// Update the scatter offsets in each load with the bin prefixes for the tile
			IterateCycles<0>::UpdateGlobalOffsets(cta, tile);

			// Scatter keys to global bin partitions
			tile->ScatterKeys(cta, valid_elements);

			// Partition values
			ScatterValues<KernelPolicy::KEYS_ONLY>::Invoke(
				cta_offset, guarded_elements, valid_elements, cta, tile);
		}
	};
*/


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

