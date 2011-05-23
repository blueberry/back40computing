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
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Abstract downsweep tile processing abstraction
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

		INVALID_BIN					= -1,
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------


	// The keys (and values) this thread will read this cycle
	KeyType 	keys[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];
	int 		key_bins[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];			// The bin for each key
	int 		key_ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];			// The tile rank of each key
	int 		counter_offsets[LOADS_PER_CYCLE][LOAD_VEC_SIZE];					// The (byte) counter offset for each key
	SizeT 		scatter_offsets[TILE_ELEMENTS_PER_THREAD];							// The global scatter offset for each key

	// Counts of my bin in each load in each cycle, valid in threads [0,BINS)
	int 		bin_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE];


	//---------------------------------------------------------------------
	// Abstract Interface
	//---------------------------------------------------------------------

	/**
	 * Returns the bin into which the specified key is to be placed.
	 *
	 * To be overloaded
	 */
	template <typename Cta>
	__device__ __forceinline__ int DecodeBin(Cta *cta, KeyType key);


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
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset);

	/**
	 * Loads keys into the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void LoadKeys(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements);


	/**
	 * Scatter keys from the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterKeys(Cta *cta);


	/**
	 * Scatter keys from the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterKeys(
		Cta *cta,
		const SizeT &guarded_elements);


	/**
	 * Scatter values from the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterValues(Cta *cta)
	{
		// Scatter values to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_values,
				values,
				scatter_offsets);
	}


	/**
	 * Scatter values from the tile
	 *
	 * To be overloaded.
	 */
	template <typename Cta>
	__device__ __forceinline__ void ScatterValues(
		Cta *cta,
		const SizeT &guarded_elements)
	{
		// Scatter values to global bin partitions
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				cta->d_out_values,
				values,
				scatter_offsets,
				guarded_elements);
	}


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Computes the number of previously-binned keys owned by the calling thread
	 * that have been marked for the specified bin.
	 */
	struct SameBinCount
	{
		// Inspect previous vec-element
		template <int CYCLE, int LOAD, int VEC>
		struct Iterate
		{
			static __device__ __forceinline__ int Invoke(Tile *tile, int current_bin)
			{
				return (current_bin == tile->key_bins[CYCLE][LOAD][VEC - 1]) +
					Iterate<CYCLE, LOAD, VEC - 1>::Invoke(tile, current_bin);
			}
		};

		// Terminate (0th vec-element has no previous elements)
		template <int CYCLE, int LOAD>
		struct Iterate<CYCLE, LOAD, 0>
		{
			static __device__ __forceinline__ int Invoke(Tile *tile, int current_bin)
			{
				return 0;
			}
		};
	};


	//---------------------------------------------------------------------
	// Cycle Methods
	//---------------------------------------------------------------------


	/**
	 * DecodeKeys
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void DecodeKeys(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch *) this;

		// Update composite-counter
		if (dispatch->template IsValid<CYCLE, LOAD, VEC>()) {

			const int PADDED_BYTES_PER_LANE 	= KernelPolicy::Grid::ROWS_PER_LANE * KernelPolicy::Grid::PADDED_PARTIALS_PER_ROW * 4;
			const int LOAD_OFFSET_BYTES 		= LOAD * KernelPolicy::SCAN_LANES_PER_LOAD * PADDED_BYTES_PER_LANE;
			const KeyType COUNTER_BYTE_MASK 	= (KernelPolicy::LOG_BINS < 2) ? 0x1 : 0x3;

			// Decode the bin for this key
			key_bins[CYCLE][LOAD][VEC] = dispatch->DecodeBin(cta, keys[CYCLE][LOAD][VEC]);

			// Decode composite-counter lane and sub-counter from bin
			int lane = key_bins[CYCLE][LOAD][VEC] >> 2;										// extract composite counter lane
			int sub_counter = key_bins[CYCLE][LOAD][VEC] & COUNTER_BYTE_MASK;				// extract 8-bit counter offset

			// Compute partial (because we overwrite, we need to accommodate all previous
			// vec-elements if they have the same bin)
			int partial = 1 + SameBinCount::template Iterate<CYCLE, LOAD, VEC>::Invoke(
				dispatch,
				key_bins[CYCLE][LOAD][VEC]);

			// Counter offset in bytes from this thread's "base_composite_counter" location
			counter_offsets[LOAD][VEC] =
				LOAD_OFFSET_BYTES +
				util::FastMul(lane, PADDED_BYTES_PER_LANE) +
				sub_counter;

			// Overwrite partial
			unsigned char *base_partial_chars = (unsigned char *) cta->base_composite_counter;
			base_partial_chars[counter_offsets[LOAD][VEC]] = partial;

		} else {

			key_bins[CYCLE][LOAD][VEC] = INVALID_BIN;
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

			unsigned char *base_partial_chars = (unsigned char *) cta->base_composite_counter;

			key_ranks[CYCLE][LOAD][VEC] = base_partial_chars[counter_offsets[LOAD][VEC]] +
				SameBinCount::template Iterate<CYCLE, LOAD, VEC>::Invoke(
					dispatch,
					key_bins[CYCLE][LOAD][VEC]);
		} else {

			// Put invalid keys just after the end of the valid swap exchange.
			key_ranks[CYCLE][LOAD][VEC] = KernelPolicy::TILE_ELEMENTS;
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
			key_ranks[CYCLE][LOAD][VEC] += cta->smem_storage.bin_prefixes[CYCLE][LOAD][key_bins[CYCLE][LOAD][VEC]];
		}
	}


	/**
	 * ResetLanes
	 */
	template <int LANE, typename Cta>
	__device__ __forceinline__ void ResetLanes(Cta *cta)
	{
		cta->base_composite_counter[LANE][0] = 0;
	}


	//---------------------------------------------------------------------
	// IterateCycleLanes Structures
	//---------------------------------------------------------------------

	/**
	 * Iterate next lane
	 */
	template <int LANE, int dummy = 0>
	struct IterateCycleLanes
	{
		// ResetLanes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ResetLanes(Cta *cta, Tile *tile)
		{
			tile->ResetLanes<LANE>(cta);
			IterateCycleLanes<LANE + 1>::ResetLanes(cta, tile);
		}
	};

	/**
	 * Terminate lane iteration
	 */
	template <int dummy>
	struct IterateCycleLanes<SCAN_LANES_PER_CYCLE, dummy>
	{
		// ResetLanes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ResetLanes(Cta *cta, Tile *tile) {}
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

		// Reset smem composite counters
		IterateCycleLanes<0>::ResetLanes(cta, dispatch);

		// Decode bins and update 8-bit composite counters for the keys in this cycle
		IterateCycleElements<CYCLE, 0, 0>::DecodeKeys(cta, dispatch);

		__syncthreads();

		// Use our raking threads to, in aggregate, scan the composite counter lanes
		if (threadIdx.x < KernelPolicy::Grid::RAKING_THREADS) {

			// Upsweep rake
			int partial = util::reduction::SerialReduce<KernelPolicy::Grid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment);

			int warpscan_lane 		= threadIdx.x >> KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE;
			int warpscan_tid 		= threadIdx.x & (KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1);

			// Inclusive warpscan in bin warpscan_lane
			int inclusive_prefix 	= util::scan::WarpScan<KernelPolicy::Grid::LOG_RAKING_THREADS_PER_LANE, false>::Invoke(
				partial,
				cta->smem_storage.lanes_warpscan[warpscan_lane],
				warpscan_tid);
			int exclusive_prefix 	= inclusive_prefix - partial;

			// Save off each lane's warpscan total for this cycle
			if (warpscan_tid == KernelPolicy::Grid::RAKING_THREADS_PER_LANE - 1) {
//				cta->smem_storage.lane_totals[CYCLE][warpscan_lane] = inclusive_prefix;
				cta->smem_storage.lane_totals[CYCLE][warpscan_lane][0] = exclusive_prefix;
				cta->smem_storage.lane_totals[CYCLE][warpscan_lane][1] = partial;
			}

			// Downsweep rake
			util::scan::SerialScan<KernelPolicy::Grid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment,
				exclusive_prefix);
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
			cta->smem_storage.lane_totals_c[CYCLE][LOAD][my_base_lane][0][my_quad_byte];
		bin_counts[CYCLE][LOAD] +=
			cta->smem_storage.lane_totals_c[CYCLE][LOAD][my_base_lane][1][my_quad_byte];

/*
		// Correct for possible overflow
		if (util::WarpVoteAll<KernelPolicy::LOG_BINS>(bin_counts[CYCLE][LOAD] <= 1)) {

			// We've potentially overflowed in this load (i.e., all keys for this
			// load have same bin.  Since keys all have the same bin, whichever bin-thread
			// has binned a key into its own bin gets all 256 counts (if that key was valid).
			Dispatch *dispatch = (Dispatch*) this;
			bin_counts[CYCLE][LOAD] = ((threadIdx.x == key_bins[CYCLE][LOAD][0]) && (dispatch->template IsValid<CYCLE, LOAD, 0>())) ?
				256 :
				0;
		}
*/
	}


	/**
	 * UpdateBinPrefixes
	 *
	 * Called by threads [0, KernelPolicy::BINS)
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void UpdateBinPrefixes(int bin_prefix, Cta *cta)
	{
		cta->smem_storage.bin_prefixes[CYCLE][LOAD][threadIdx.x] = bin_counts[CYCLE][LOAD] + bin_prefix;
	}


	/**
	 * ComputeScatterOffsets
	 */
	template <int ELEMENT, typename Cta>
	__device__ __forceinline__ void ComputeScatterOffsets(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		KeyType *linear_keys = (KeyType *) keys;
		int bin = dispatch->DecodeBin(cta, linear_keys[ELEMENT]);
		scatter_offsets[ELEMENT] = cta->smem_storage.bin_carry[bin] + (KernelPolicy::THREADS * ELEMENT) + threadIdx.x;
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
		// ComputeScatterOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeScatterOffsets(Cta *cta, Tile *tile)
		{
			tile->ComputeScatterOffsets<ELEMENT>(cta);
			IterateElements<ELEMENT + 1>::ComputeScatterOffsets(cta, tile);
		}
	};


	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateElements<TILE_ELEMENTS_PER_THREAD, dummy>
	{
		// ComputeScatterOffsets
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void ComputeScatterOffsets(Cta *cta, Tile *tile) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Compute global scatter offsets for gathered keys
	 */
	template <typename Cta>
	__device__ __forceinline__ void ComputeScatterOffsets(Cta *cta)
	{
		IterateElements<0>::ComputeScatterOffsets(cta, (Dispatch*) this);
	}


	/**
	 * Partition keys
	 */
	template <typename Cta>
	__device__ __forceinline__ void PartitionKeys(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Scan cycles
		IterateCycles<0>::ScanCycles(cta, dispatch);

		// Scan across bins
		if (threadIdx.x < KernelPolicy::BINS) {

			// Recover bin-counts from lane totals
			int my_base_lane = threadIdx.x >> 2;
			int my_quad_byte = threadIdx.x & 3;
			IterateCycleLoads<0, 0>::RecoverBinCounts(my_base_lane, my_quad_byte, cta, dispatch);

			// Scan across my bin counts for each load
			int tile_bin_total = util::scan::SerialScan<KernelPolicy::LOADS_PER_TILE>::Invoke(
				(int *) bin_counts, 0);

			// Add the previous tile's inclusive-scan to the running bin-carry
			SizeT my_carry = cta->smem_storage.bin_carry[threadIdx.x] + cta->smem_storage.bin_warpscan[1][threadIdx.x];

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
			IterateCycleLoads<0, 0>::UpdateBinPrefixes(tile_bin_exclusive, cta, dispatch);
		}

		__syncthreads();

		// Update the key ranks in each load with the bin prefixes for the tile
		IterateCycles<0>::UpdateRanks(cta, dispatch);

		// Scatter keys to smem
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			util::io::st::NONE>::Scatter(
				cta->smem_storage.smem_pool.key_exchange,
				(KeyType *) keys,
				(int *) key_ranks);

		__syncthreads();

		// Gather keys from smem (vec-1)
		util::io::LoadTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelPolicy::THREADS,
			util::io::ld::NONE>::LoadValid(
				(KeyType (*)[1]) keys,
				cta->smem_storage.smem_pool.key_exchange);

		__syncthreads();
	}


	/**
	 * Partition values
	 */
	template <typename Cta>
	__device__ __forceinline__ void PartitionValues(Cta *cta)
	{
		// Scatter values to smem
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			util::io::st::NONE>::Scatter(
				cta->smem_storage.smem_pool.value_exchange,
				values,
				(int *) key_ranks);

		__syncthreads();

		// Gather values from smem (vec-1)
		util::io::LoadTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelPolicy::THREADS,
			util::io::ld::NONE>::LoadValid(
				(ValueType (*)[1]) values,
				cta->smem_storage.smem_pool.value_exchange);

		__syncthreads();
	}
};


} // namespace downsweep
} // namespace partition
} // namespace b40c

