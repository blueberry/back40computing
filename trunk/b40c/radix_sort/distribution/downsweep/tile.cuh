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
 * AUTHORS' REQUEST: 
 * 
 * 		If you use|reference|benchmark this code, please cite our Technical 
 * 		Report (http://www.cs.virginia.edu/~dgm4d/papers/RadixSortTR.pdf):
 * 
 *		@TechReport{ Merrill:Sorting:2010,
 *        	author = "Duane Merrill and Andrew Grimshaw",
 *        	title = "Revisiting Sorting for GPGPU Stream Architectures",
 *        	year = "2010",
 *        	institution = "University of Virginia, Department of Computer Science",
 *        	address = "Charlottesville, VA, USA",
 *        	number = "CS2010-03"
 *		}
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 * Upsweep CTA tile processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>
#include <b40c/util/device_intrinsics.cuh>

#include <b40c/radix_sort/sort_utils.cuh>

namespace b40c {
namespace radix_sort {
namespace distribution {
namespace downsweep {


/**
 * Tile
 */
template <
	typename KernelConfig,
	typename Derived = util::NullType>
struct Tile
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelConfig::KeyType 					KeyType;
	typedef typename KernelConfig::ValueType 				ValueType;
	typedef typename KernelConfig::SizeT 					SizeT;

	typedef typename util::If<util::Equals<Derived, util::NullType>::VALUE, Tile, Derived>::Type Dispatch;

	enum {
		LOAD_VEC_SIZE 				= KernelConfig::LOAD_VEC_SIZE,
		LOADS_PER_CYCLE 			= KernelConfig::LOADS_PER_CYCLE,
		CYCLES_PER_TILE 			= KernelConfig::CYCLES_PER_TILE,
		TILE_ELEMENTS_PER_THREAD 	= KernelConfig::TILE_ELEMENTS_PER_THREAD,
		SCAN_LANES_PER_CYCLE		= KernelConfig::SCAN_LANES_PER_CYCLE,
	};

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------


	KeyType 	keys[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];					// The keys this thread will read this cycle
	ValueType 	values[TILE_ELEMENTS_PER_THREAD];

	int 		key_digits[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];			// Their decoded digits
	int 		key_ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE][LOAD_VEC_SIZE];				// The tile rank of each key

	int 		counter_offsets[LOADS_PER_CYCLE][LOAD_VEC_SIZE];

	int 		digit_counts[CYCLES_PER_TILE][LOADS_PER_CYCLE];							// Counts of my digit in each load in each cycle

	SizeT 		scatter_offsets[TILE_ELEMENTS_PER_THREAD];

	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	struct SameDigitCount
	{
		// Inspect previous vec-element
		template <int CYCLE, int LOAD, int VEC>
		struct Iterate
		{
			static __device__ __forceinline__ int Invoke(Tile *tile, int current_digit)
			{
				return (current_digit == tile->key_digits[CYCLE][LOAD][VEC - 1]) +
					Iterate<CYCLE, LOAD, VEC - 1>::Invoke(tile, current_digit);
			}
		};

		// Terminate (0th vec-element has no previous elements)
		template <int CYCLE, int LOAD>
		struct Iterate<CYCLE, LOAD, 0>
		{
			static __device__ __forceinline__ int Invoke(Tile *tile, int current_digit)
			{
				return 0;
			}
		};
	};


	//---------------------------------------------------------------------
	// Cycle Internal Methods
	//---------------------------------------------------------------------

	/**
	 * DecodeKeys
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void DecodeKeys(Cta *cta)
	{
		const int PADDED_BYTES_PER_LANE 	= KernelConfig::Grid::ROWS_PER_LANE * KernelConfig::Grid::PADDED_PARTIALS_PER_ROW * 4;
		const int LOAD_OFFSET_BYTES 		= LOAD * KernelConfig::SCAN_LANES_PER_LOAD * PADDED_BYTES_PER_LANE;
		const KeyType COUNTER_BYTE_MASK 	= (KernelConfig::RADIX_BITS < 2) ? 0x1 : 0x3;

		key_digits[CYCLE][LOAD][VEC] = cta->DecodeDigit(keys[CYCLE][LOAD][VEC]);			// extract digit

		int lane = key_digits[CYCLE][LOAD][VEC] >> 2;										// extract composite counter lane
		int counter_byte = key_digits[CYCLE][LOAD][VEC] & COUNTER_BYTE_MASK;				// extract 8-bit counter offset

		// Compute partial (because we overwrite, we need to accommodate all previous
		// vec-elements if they have the same digit)
		int partial = 1 + SameDigitCount::template Iterate<CYCLE, LOAD, VEC>::Invoke(
			(Dispatch*) this,
			key_digits[CYCLE][LOAD][VEC]);

		// Counter offset in bytes from this thread's "base_partial" location
		counter_offsets[LOAD][VEC] = LOAD_OFFSET_BYTES + util::FastMul(lane, PADDED_BYTES_PER_LANE) + counter_byte;

		// Overwrite partial
		unsigned char *base_partial_chars = (unsigned char *) cta->base_partial;
		base_partial_chars[counter_offsets[LOAD][VEC]] = partial;
	}


	/**
	 * ExtractRanks
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void ExtractRanks(Cta *cta)
	{
		unsigned char *base_partial_chars = (unsigned char *) cta->base_partial;

		key_ranks[CYCLE][LOAD][VEC] = base_partial_chars[counter_offsets[LOAD][VEC]] +
			SameDigitCount::template Iterate<CYCLE, LOAD, VEC>::Invoke(
				(Dispatch*) this,
				key_digits[CYCLE][LOAD][VEC]);
	}


	/**
	 * UpdateRanks
	 */
	template <int CYCLE, int LOAD, int VEC, typename Cta>
	__device__ __forceinline__ void UpdateRanks(Cta *cta)
	{
		key_ranks[CYCLE][LOAD][VEC] += cta->smem_storage.digit_prefixes[CYCLE][LOAD][key_digits[CYCLE][LOAD][VEC]];
	}


	/**
	 * ResetLanes
	 */
	template <int LANE, typename Cta>
	__device__ __forceinline__ void ResetLanes(Cta *cta)
	{
		cta->base_partial[LANE][0] = 0;
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

		// Decode digits and update 8-bit composite counters for the keys in this cycle
		IterateCycleElements<CYCLE, 0, 0>::DecodeKeys(cta, dispatch);

		__syncthreads();

		// Use our raking threads to, in aggregate, scan the composite counter lanes
		if (threadIdx.x < KernelConfig::Grid::RAKING_THREADS) {

			// Upsweep rake
			int partial = util::reduction::SerialReduce<KernelConfig::Grid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment);

			// Inclusive warpscan in digit warpscan_lane
			int warpscan_lane 		= threadIdx.x >> KernelConfig::Grid::LOG_RAKING_THREADS_PER_LANE;
			int warpscan_tid 		= threadIdx.x & (KernelConfig::Grid::RAKING_THREADS_PER_LANE - 1);
			int inclusive_prefix 	= util::scan::WarpScan<KernelConfig::Grid::LOG_RAKING_THREADS_PER_LANE, false>::Invoke(
				partial,
				cta->smem_storage.lanes_warpscan[warpscan_lane],
				warpscan_tid);
			int exclusive_prefix 	= inclusive_prefix - partial;

			// Save off each lane's warpscan total for this cycle
			if (warpscan_tid == KernelConfig::Grid::RAKING_THREADS_PER_LANE - 1) {
				cta->smem_storage.lane_totals[CYCLE][warpscan_lane] = inclusive_prefix;
			}

			// Downsweep rake
			util::scan::SerialScan<KernelConfig::Grid::PARTIALS_PER_SEG>::Invoke(
				cta->raking_segment,
				exclusive_prefix);
		}

		__syncthreads();

		// Extract the local ranks of each key
		IterateCycleElements<CYCLE, 0, 0>::ExtractRanks(cta, dispatch);
	}


	/**
	 * RecoverDigitCounts
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void RecoverDigitCounts(
		int my_base_lane, int my_quad_byte, Cta *cta)
	{
		unsigned char *composite_counter = (unsigned char *)
			&cta->smem_storage.lane_totals[CYCLE][my_base_lane + (KernelConfig::SCAN_LANES_PER_LOAD * LOAD)];

		digit_counts[CYCLE][LOAD] = composite_counter[my_quad_byte];

		// Correct for possible overflow
		if (util::WarpVoteAll<KernelConfig::RADIX_BITS>(digit_counts[CYCLE][LOAD] <= 1)) {

			// We overflowed in this load: all keys for this load have same
			// digit, i.e., whoever owns the digit matching one of their own
			// key's digit gets all 256 counts
			digit_counts[CYCLE][LOAD] = (threadIdx.x == key_digits[CYCLE][LOAD][0]) ? 256 : 0;
		}
	}


	/**
	 * UpdateDigitPrefixes
	 */
	template <int CYCLE, int LOAD, typename Cta>
	__device__ __forceinline__ void UpdateDigitPrefixes(int digit_prefix, Cta *cta)
	{
		cta->smem_storage.digit_prefixes[CYCLE][LOAD][threadIdx.x] = digit_counts[CYCLE][LOAD] + digit_prefix;
	}


	/**
	 * ComputeScatterOffsets
	 */
	template <int ELEMENT, typename Cta>
	__device__ __forceinline__ void ComputeScatterOffsets(Cta *cta)
	{
		KeyType *linear_keys = (KeyType *) keys;
		int digit = cta->DecodeDigit(linear_keys[ELEMENT]);
		scatter_offsets[ELEMENT] = cta->smem_storage.digit_carry[digit] + (KernelConfig::THREADS * ELEMENT) + threadIdx.x;
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
		// RecoverDigitCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverDigitCounts(
			int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile)
		{
			tile->RecoverDigitCounts<CYCLE, LOAD>(my_base_lane, my_quad_byte, cta);
			IterateCycleLoads<CYCLE, LOAD + 1>::RecoverDigitCounts(my_base_lane, my_quad_byte, cta, tile);
		}

		// UpdateDigitPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateDigitPrefixes(
			int digit_prefix, Cta *cta, Tile *tile)
		{
			tile->UpdateDigitPrefixes<CYCLE, LOAD>(digit_prefix, cta);
			IterateCycleLoads<CYCLE, LOAD + 1>::UpdateDigitPrefixes(digit_prefix, cta, tile);
		}
	};


	/**
	 * Iterate next cycle
	 */
	template <int CYCLE, int dummy>
	struct IterateCycleLoads<CYCLE, LOADS_PER_CYCLE, dummy>
	{
		// RecoverDigitCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverDigitCounts(
			int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile)
		{
			IterateCycleLoads<CYCLE + 1, 0>::RecoverDigitCounts(my_base_lane, my_quad_byte, cta, tile);
		}

		// UpdateDigitPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateDigitPrefixes(
			int digit_prefix, Cta *cta, Tile *tile)
		{
			IterateCycleLoads<CYCLE + 1, 0>::UpdateDigitPrefixes(digit_prefix, cta, tile);
		}
	};

	/**
	 * Terminate iteration
	 */
	template <int dummy>
	struct IterateCycleLoads<CYCLES_PER_TILE, 0, dummy>
	{
		// RecoverDigitCounts
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void RecoverDigitCounts(int my_base_lane, int my_quad_byte, Cta *cta, Tile *tile) {}

		// UpdateDigitPrefixes
		template <typename Cta, typename Tile>
		static __device__ __forceinline__ void UpdateDigitPrefixes(int digit_prefix, Cta *cta, Tile *tile) {}
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
	 * Sort keys
	 */
	template <typename Cta>
	__device__ __forceinline__ void SortKeys(Cta *cta)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Scan cycles
		IterateCycles<0>::ScanCycles(cta, dispatch);

		// Scan across digits
		if (threadIdx.x < KernelConfig::RADIX_DIGITS) {

			// Recover digit-counts from lanes_warpscan padding
			int my_base_lane = threadIdx.x >> 2;
			int my_quad_byte = threadIdx.x & 3;
			IterateCycleLoads<0, 0>::RecoverDigitCounts(my_base_lane, my_quad_byte, cta, dispatch);

			// Scan across my digit counts for each load
			int inclusive_total = util::scan::SerialScan<KernelConfig::LOADS_PER_TILE>::Invoke(
				(int *) digit_counts, 0);

			// Add the inclusive scan of digit counts from the previous tile to the running carry
			SizeT my_carry = cta->smem_storage.digit_carry[threadIdx.x] + cta->smem_storage.digit_warpscan[1][threadIdx.x];

			// Perform overflow-free inclusive SIMD Kogge-Stone across digits
			int digit_prefix_inclusive = util::scan::WarpScan<KernelConfig::RADIX_BITS, false>::Invoke(
				inclusive_total,
				cta->smem_storage.digit_warpscan);

			// Save inclusive scan in digit_warpscan
			cta->smem_storage.digit_warpscan[1][threadIdx.x] = digit_prefix_inclusive;

			// Calculate exclusive scan
			int digit_prefix_exclusive = digit_prefix_inclusive - inclusive_total;

			// Subtract the digit prefix from the running carry (to offset threadIdx during scatter)
			cta->smem_storage.digit_carry[threadIdx.x] = my_carry - digit_prefix_exclusive;

			// Compute the digit prefixes for this tile for each load
			IterateCycleLoads<0, 0>::UpdateDigitPrefixes(digit_prefix_exclusive, cta, dispatch);
		}

		__syncthreads();

		// Update the key ranks in each load with the digit prefixes for the tile
		IterateCycles<0>::UpdateRanks(cta, dispatch);

		// Scatter keys to smem
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			util::io::st::NONE>::Scatter(
				cta->smem_storage.smem_pool.key_exchange,
				(KeyType *) keys,
				(int *) key_ranks);

		__syncthreads();

		// Gather keys from smem (vec-1)
		util::io::LoadTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelConfig::THREADS,
			util::io::ld::NONE>::LoadValid(
				(KeyType (*)[1]) keys,
				cta->smem_storage.smem_pool.key_exchange);
	}


	/**
	 * Sort values
	 */
	template <typename Cta>
	__device__ __forceinline__ void SortValues(Cta *cta)
	{
		// Scatter values to smem
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			util::io::st::NONE>::Scatter(
				cta->smem_storage.smem_pool.value_exchange,
				values,
				(int *) key_ranks);

		__syncthreads();

		// Gather values from smem (vec-1)
		util::io::LoadTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0,
			KernelConfig::THREADS,
			util::io::ld::NONE>::LoadValid(
				(ValueType (*)[1]) values,
				cta->smem_storage.smem_pool.value_exchange);
	}


	/**
	 * Process full tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void Process(
		Cta *cta,
		SizeT cta_offset)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Read tile of keys, use -1 if key is out-of-bounds
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE, 				// Number of vector loads (log)
			KernelConfig::LOG_LOAD_VEC_SIZE,				// Number of items per vector load (log)
			KernelConfig::THREADS,							// Active threads that will be loading
			KernelConfig::READ_MODIFIER>					// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
				::template LoadValid<KeyType, KernelConfig::PreprocessTraits::Preprocess>(
					(KeyType (*)[KernelConfig::LOAD_VEC_SIZE]) keys,
					cta->d_in_keys + cta_offset);

		// Sort keys
		SortKeys(cta);

		__syncthreads();

		// Compute global scatter offsets for gathered keys
		IterateElements<0>::ComputeScatterOffsets(cta, dispatch);

		// Scatter keys to global digit partitions
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER>::template Scatter<
				KeyType,
				KernelConfig::PostprocessTraits::Postprocess>(
					cta->d_out_keys,
					(KeyType *) keys,
					scatter_offsets);

		if (!util::Equals<ValueType, util::NullType>::VALUE) {

			// Read values
			util::io::LoadTile<
				KernelConfig::LOG_LOADS_PER_TILE, 				// Number of vector loads (log)
				KernelConfig::LOG_LOAD_VEC_SIZE,				// Number of items per vector load (log)
				KernelConfig::THREADS,							// Active threads that will be loading
				KernelConfig::READ_MODIFIER>::LoadValid(		// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
					(ValueType (*)[KernelConfig::LOAD_VEC_SIZE]) values,
					cta->d_in_values + cta_offset);

			// Sort values
			SortValues(cta);

			__syncthreads();

			// Scatter values to global digit partitions
			util::io::ScatterTile<
				KernelConfig::TILE_ELEMENTS_PER_THREAD,
				KernelConfig::THREADS,
				KernelConfig::WRITE_MODIFIER>::Scatter(
					cta->d_out_values,
					values,
					scatter_offsets);
		}

	}



	/**
	 * Process partial tile
	 */
	template <typename Cta>
	__device__ __forceinline__ void Process(
		Cta *cta,
		SizeT cta_offset,
		const SizeT &guarded_elements)
	{
		Dispatch *dispatch = (Dispatch*) this;

		// Read tile of keys, use -1 if key is out-of-bounds
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE, 				// Number of vector loads (log)
			KernelConfig::LOG_LOAD_VEC_SIZE,				// Number of items per vector load (log)
			KernelConfig::THREADS,							// Active threads that will be loading
			KernelConfig::READ_MODIFIER>					// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
				::template LoadValid<KeyType, KernelConfig::PreprocessTraits::Preprocess>(
					(KeyType (*)[KernelConfig::LOAD_VEC_SIZE]) keys,
					(KeyType) -1,
					cta->d_in_keys + cta_offset,
					guarded_elements);

		// Sort keys
		SortKeys(cta);

		__syncthreads();

		// Compute global scatter offsets for gathered keys
		IterateElements<0>::ComputeScatterOffsets(cta, dispatch);

		// Scatter keys to global digit partitions
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER>::template Scatter<
				KeyType,
				KernelConfig::PostprocessTraits::Postprocess>(
					cta->d_out_keys,
					(KeyType *) keys,
					scatter_offsets,
					guarded_elements);

		if (!util::Equals<ValueType, util::NullType>::VALUE) {

			// Read values
			util::io::LoadTile<
				KernelConfig::LOG_LOADS_PER_TILE, 				// Number of vector loads (log)
				KernelConfig::LOG_LOAD_VEC_SIZE,				// Number of items per vector load (log)
				KernelConfig::THREADS,							// Active threads that will be loading
				KernelConfig::READ_MODIFIER>::LoadValid(		// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
					(ValueType (*)[KernelConfig::LOAD_VEC_SIZE]) values,
					cta->d_in_values + cta_offset);

			// Sort values
			SortValues(cta);

			__syncthreads();

			// Scatter values to global digit partitions
			util::io::ScatterTile<
				KernelConfig::TILE_ELEMENTS_PER_THREAD,
				KernelConfig::THREADS,
				KernelConfig::WRITE_MODIFIER>::Scatter(
					cta->d_out_values,
					values,
					scatter_offsets,
					guarded_elements);
		}

	}
};


} // namespace downsweep
} // namespace distribution
} // namespace radix_sort
} // namespace b40c

