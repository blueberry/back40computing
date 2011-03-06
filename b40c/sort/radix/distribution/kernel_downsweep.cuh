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
 * Downsweep scan-scatter kernel.  The third kernel in a radix-sorting 
 * digit-place pass.
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_utils.cuh"
#include "b40c_kernel_data_movement.cuh"
#include "radixsort_common.cuh"

namespace b40c {
namespace radix_sort {
namespace downsweep {


/******************************************************************************
 * Granularity Configuration
 ******************************************************************************/

/**
 * Downsweep granularity configuration.  This C++ type encapsulates our 
 * kernel-tuning parameters (they are reflected via the static fields).
 *  
 * The kernels are specialized for problem-type, SM-version, etc. by declaring 
 * them with different performance-tuned parameterizations of this type.  By 
 * incorporating this type into the kernel code itself, we guide the compiler in 
 * expanding/unrolling the kernel code for specific architectures and problem 
 * types.
 * 
 * Constraints:
 * 		(i) 	A load can't contain more than 256 keys or we might overflow inside a lane of  
 * 				8-bit composite counters, i.e., (threads * load-vec-size <= 256), equivalently:
 * 
 * 					(LOG_THREADS + LOG_LOAD_VEC_SIZE <= 8)
 * 
 * 		(ii) 	We must have between one and one warp of raking threads per lane of composite 
 * 				counters, i.e., (1 <= raking-threads / (loads-per-cycle * radix-digits / 4) <= 32), 
 * 				equivalently:
 * 
 * 					(0 <= LOG_RAKING_THREADS - LOG_LOADS_PER_CYCLE - RADIX_BITS + 2 <= B40C_LOG_WARP_THREADS(arch))
 *     
 * 		(iii) 	We must have more than radix-digits threads in the threadblock,  
 * 				i.e., (threads >= radix-digits) equivalently:
 * 
 * 					LOG_THREADS >= RADIX_BITS
 */
template <
	typename _KeyType,
	typename _ValueType,
	typename _SizeT,
	int _RADIX_BITS,
	int _LOG_SCHEDULE_GRANULARITY,
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_CYCLE,
	int _LOG_CYCLES_PER_TILE,
	int _LOG_RAKING_THREADS,
	CacheModifier _CACHE_MODIFIER,
	bool _EARLY_EXIT>

struct DownsweepConfig
{
	typedef _KeyType							KeyType;
	typedef _ValueType							ValueType;
	typedef _SizeT								SizeT;
	static const int RADIX_BITS					= _RADIX_BITS;
	static const int LOG_SCHEDULE_GRANULARITY	= _LOG_SCHEDULE_GRANULARITY;
	static const int CTA_OCCUPANCY  			= _CTA_OCCUPANCY;
	static const int LOG_THREADS 				= _LOG_THREADS;
	static const int LOG_LOAD_VEC_SIZE 			= _LOG_LOAD_VEC_SIZE;
	static const int LOG_LOADS_PER_CYCLE		= _LOG_LOADS_PER_CYCLE;
	static const int LOG_CYCLES_PER_TILE		= _LOG_CYCLES_PER_TILE;
	static const int LOG_RAKING_THREADS			= _LOG_RAKING_THREADS;
	static const CacheModifier CACHE_MODIFIER 	= _CACHE_MODIFIER;
	static const bool EARLY_EXIT				= _EARLY_EXIT;
};



/******************************************************************************
 * Kernel Configuration  
 ******************************************************************************/

/**
 * A detailed downsweep configuration type that specializes kernel code for a 
 * specific sorting pass.  It encapsulates granularity details derived from the 
 * inherited DownsweepConfigType
 */
template <
	typename 		DownsweepConfigType,
	typename 		PreprocessTraitsType, 
	typename 		PostprocessTraitsType, 
	int 			_CURRENT_PASS,
	int 			_CURRENT_BIT>
struct DownsweepKernelConfig : DownsweepConfigType
{
	typedef PreprocessTraitsType					PreprocessTraits;
	typedef PostprocessTraitsType					PostprocessTraits;

	static const int RADIX_DIGITS 					= 1 << DownsweepConfigType::RADIX_BITS;
	static const int CURRENT_PASS					= _CURRENT_PASS;
	static const int CURRENT_BIT					= _CURRENT_BIT;

	static const int THREADS						= 1 << DownsweepConfigType::LOG_THREADS;
	
	static const int LOG_WARPS						= DownsweepConfigType::LOG_THREADS - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__);
	static const int WARPS							= 1 << LOG_WARPS;	
	
	static const int LOAD_VEC_SIZE					= 1 << DownsweepConfigType::LOG_LOAD_VEC_SIZE;
	static const int LOADS_PER_CYCLE				= 1 << DownsweepConfigType::LOG_LOADS_PER_CYCLE;
	static const int CYCLES_PER_TILE				= 1 << DownsweepConfigType::LOG_CYCLES_PER_TILE;
	
	static const int LOG_LOADS_PER_TILE				= DownsweepConfigType::LOG_LOADS_PER_CYCLE + 
														DownsweepConfigType::LOG_CYCLES_PER_TILE;
	static const int LOADS_PER_TILE					= 1 << LOG_LOADS_PER_TILE;
	
	static const int LOG_CYCLE_ELEMENTS				= DownsweepConfigType::LOG_THREADS +
														DownsweepConfigType::LOG_LOADS_PER_CYCLE +
														DownsweepConfigType::LOG_LOAD_VEC_SIZE;
	static const int CYCLE_ELEMENTS					= 1 << LOG_CYCLE_ELEMENTS;
	
	static const int LOG_TILE_ELEMENTS				= DownsweepConfigType::LOG_CYCLES_PER_TILE + LOG_CYCLE_ELEMENTS;
	static const int TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS;
	
	static const int LOG_TILE_ELEMENTS_PER_THREAD	= LOG_TILE_ELEMENTS - DownsweepConfigType::LOG_THREADS;
	static const int TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD;

	static const int LOG_SCAN_LANES_PER_LOAD		= B40C_MAX((DownsweepConfigType::RADIX_BITS - 2), 0);		// Always at least one lane per load
	static const int SCAN_LANES_PER_LOAD			= 1 << LOG_SCAN_LANES_PER_LOAD;								
	
	static const int LOG_SCAN_LANES_PER_CYCLE		= DownsweepConfigType::LOG_LOADS_PER_CYCLE + LOG_SCAN_LANES_PER_LOAD;
	static const int SCAN_LANES_PER_CYCLE			= 1 << LOG_SCAN_LANES_PER_CYCLE;

	// Smem SRTS grid type for reducing and scanning a cycle of 
	// (radix-digits/4) lanes of composite 8-bit digit counters
	typedef SrtsGrid<
		int,											// type
		DownsweepConfigType::LOG_THREADS,				// depositing threads (lane size)
		LOG_SCAN_LANES_PER_CYCLE, 						// lanes
		DownsweepConfigType::LOG_RAKING_THREADS> 		// raking threads
			Grid;
	
	static const int EXCHANGE_BYTES					= B40C_MAX(
														(TILE_ELEMENTS * sizeof(DownsweepConfigType::KeyType)), 
														(TILE_ELEMENTS * sizeof(DownsweepConfigType::ValueType)));
	
	static const int SMEM_POOL_BYTES				= B40C_MAX(Grid::SMEM_BYTES, EXCHANGE_BYTES);

	// Must allocate in 64-bit chunks to ensure correct alignment of arbitrary value-types
	static const int SMEM_POOL_INT4S         		= (SMEM_POOL_BYTES + sizeof(int4) - 1) / sizeof(int4);
};
	


/******************************************************************************
 * Tile-processing Routines
 ******************************************************************************/

template <typename Config> 
__device__ __forceinline__ int DecodeDigit(typename Config::KeyType key) 
{
	int retval;
	ExtractKeyBits<typename Config::KeyType, Config::CURRENT_BIT, Config::RADIX_BITS>::Extract(retval, key);
	return retval;
}


// Count previous keys in the vector having the same digit as current_digit
template <typename Config>
struct SameDigitCount
{
	typedef typename Config::KeyType KeyType;

	// Inspect prev vec-element
	template <int LOAD, int VEC_ELEMENT>
	struct Iterate
	{
		static __device__ __forceinline__ int Invoke(
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int current_digit) 
		{
			return (current_digit == key_digits[LOAD][VEC_ELEMENT - 1]) + Iterate<LOAD, VEC_ELEMENT - 1>::Invoke(key_digits, current_digit);
		}
	};
	
	// Terminate (0th vec-element has no previous elements) 
	template <int LOAD>
	struct Iterate<LOAD, 0>
	{
		static __device__ __forceinline__ int Invoke(
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int current_digit) 
		{
			return 0;
		}
	};
};


// Decode a cycle of keys
template <typename Config> 
struct DecodeCycleKeys
{
	typedef typename Config::KeyType KeyType;
	
	// Next vec-element
	template <int LOAD, int TOTAL_LOADS, int VEC_ELEMENT, int TOTAL_VEC_ELEMENTS>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			KeyType keys[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int *base_partial) 
		{
			const int PADDED_BYTES_PER_LANE 	= Config::Grid::ROWS_PER_LANE * Config::Grid::PADDED_PARTIALS_PER_ROW * 4;
			const int LOAD_OFFSET_BYTES 		= LOAD * Config::SCAN_LANES_PER_LOAD * PADDED_BYTES_PER_LANE;
			const KeyType COUNTER_BYTE_MASK 	= (Config::RADIX_BITS < 2) ? 0x1 : 0x3;
			
			key_digits[LOAD][VEC_ELEMENT] = DecodeDigit<Config>(keys[LOAD][VEC_ELEMENT]);			// extract digit
			int lane = key_digits[LOAD][VEC_ELEMENT] >> 2;									// extract composite counter lane
			int counter_byte = key_digits[LOAD][VEC_ELEMENT] & COUNTER_BYTE_MASK;			// extract 8-bit counter offset

			// Compute partial (because we overwrite, we need to accommodate all previous vec-elements if they have the same digit)
			int partial = 1 + SameDigitCount<Config>::template Iterate<LOAD, VEC_ELEMENT>::Invoke(key_digits, key_digits[LOAD][VEC_ELEMENT]);

			// Counter offset in bytes from this thread's "base_partial" location
			counter_offsets[LOAD][VEC_ELEMENT] = LOAD_OFFSET_BYTES + FastMul(lane, PADDED_BYTES_PER_LANE) + counter_byte;

			// Overwrite partial
			unsigned char *base_partial_chars = reinterpret_cast<unsigned char *>(base_partial);
			base_partial_chars[counter_offsets[LOAD][VEC_ELEMENT]] = partial;
			
			// Next
			Iterate<LOAD, TOTAL_LOADS, VEC_ELEMENT + 1, TOTAL_VEC_ELEMENTS>::Invoke(
				keys, key_digits, counter_offsets, base_partial);
		}
	};

	// Next Load
	template <int LOAD, int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<LOAD, TOTAL_LOADS, TOTAL_VEC_ELEMENTS, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			KeyType keys[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int *base_partial) 
		{
			Iterate<LOAD + 1, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>::Invoke(
				keys, key_digits, counter_offsets, base_partial);
		}
	};
	
	// Terminate
	template <int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			KeyType keys[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int *base_partial) {} 
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		KeyType keys[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
		int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int *base_partial) 
	{
		Iterate<0, Config::LOADS_PER_CYCLE, 0, Config::LOAD_VEC_SIZE>::Invoke(
			keys, key_digits, counter_offsets, base_partial);
	} 
};


// Extract cycle ranks
template <typename Config> 
struct ExtractCycleRanks
{
	// Next vec-element
	template <int LOAD, int TOTAL_LOADS, int VEC_ELEMENT, int TOTAL_VEC_ELEMENTS>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int *base_partial) 
		{
			unsigned char *base_partial_chars = reinterpret_cast<unsigned char *>(base_partial);

			key_ranks[LOAD][VEC_ELEMENT] = base_partial_chars[counter_offsets[LOAD][VEC_ELEMENT]] +
				SameDigitCount<Config>::template Iterate<LOAD, VEC_ELEMENT>::Invoke(key_digits, key_digits[LOAD][VEC_ELEMENT]);

			// Next
			Iterate<LOAD, TOTAL_LOADS, VEC_ELEMENT + 1, TOTAL_VEC_ELEMENTS>::Invoke(
				key_ranks, key_digits, counter_offsets, base_partial);
		}
	};

	// Next Load
	template <int LOAD, int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<LOAD, TOTAL_LOADS, TOTAL_VEC_ELEMENTS, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int *base_partial) 
		{
			Iterate<LOAD + 1, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>::Invoke(key_ranks, key_digits, counter_offsets, base_partial);
		}
	};
	
	// Terminate
	template <int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int *base_partial) {} 
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		int key_ranks[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
		int key_digits[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int *base_partial) 
	{
		Iterate<0, Config::LOADS_PER_CYCLE, 0, Config::LOAD_VEC_SIZE>::Invoke(
			key_ranks, key_digits, counter_offsets, base_partial);
	} 
};


// Scan a cycle of lanes
// Called by threads [0 .. raking-threads- 1]
template <typename Config>
__device__ __forceinline__ void ScanCycleLanes(
	int* 	raking_segment,
	int 	lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE],
	int 	lane_totals[Config::SCAN_LANES_PER_CYCLE])
{
	// Upsweep rake
	int partial = SerialReduce<int, Config::Grid::PARTIALS_PER_SEG>::Invoke(raking_segment);

	// Warpscan reduction in digit warpscan_lane
	int warpscan_lane = threadIdx.x >> Config::Grid::LOG_RAKING_THREADS_PER_LANE;
	int warpscan_tid = threadIdx.x & (Config::Grid::RAKING_THREADS_PER_LANE - 1);

	int inclusive_prefix = WarpScanInclusive<int, Config::Grid::LOG_RAKING_THREADS_PER_LANE>::Invoke(
		partial, lanes_warpscan[warpscan_lane], warpscan_tid);
	int exclusive_prefix = inclusive_prefix - partial;
	
	// Save off each lane's warpscan total for this cycle
	if (warpscan_tid == Config::Grid::RAKING_THREADS_PER_LANE - 1) lane_totals[warpscan_lane] = inclusive_prefix;

	// Downsweep rake
	SerialScan<int, Config::Grid::PARTIALS_PER_SEG>::Invoke(raking_segment, exclusive_prefix);
}


// Update ranks in each load with digit prefixes for the current tile
template <typename Config> 
struct UpdateTileRanks
{
	// Next vec-element
	template <int CYCLE, int TOTAL_CYCLES, int LOAD, int TOTAL_LOADS, int VEC_ELEMENT, int TOTAL_VEC_ELEMENTS>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS])
		{
			key_ranks[CYCLE][LOAD][VEC_ELEMENT] += digit_prefixes[CYCLE][LOAD][key_digits[CYCLE][LOAD][VEC_ELEMENT]];
			Iterate<CYCLE, TOTAL_CYCLES, LOAD, TOTAL_LOADS, VEC_ELEMENT + 1, TOTAL_VEC_ELEMENTS>::Invoke(
				key_ranks, key_digits, digit_prefixes);
		}
	};

	// Next Load
	template <int CYCLE, int TOTAL_CYCLES, int LOAD, int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<CYCLE, TOTAL_CYCLES, LOAD, TOTAL_LOADS, TOTAL_VEC_ELEMENTS, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS]) 
		{
			Iterate<CYCLE, TOTAL_CYCLES, LOAD + 1, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>::Invoke(
				key_ranks, key_digits, digit_prefixes);
		}
	};

	// Next Cycle
	template <int CYCLE, int TOTAL_CYCLES, int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<CYCLE, TOTAL_CYCLES, TOTAL_LOADS, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS])
		{
			Iterate<CYCLE + 1, TOTAL_CYCLES, 0, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>::Invoke(
				key_ranks, key_digits, digit_prefixes);
		}
	};

	// Terminate
	template <int TOTAL_CYCLES, int TOTAL_LOADS, int TOTAL_VEC_ELEMENTS>
	struct Iterate<TOTAL_CYCLES, TOTAL_CYCLES, 0, TOTAL_LOADS, 0, TOTAL_VEC_ELEMENTS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS]) {} 
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		int key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE], 
		int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS]) 
	{
		Iterate<0, Config::CYCLES_PER_TILE, 0, Config::LOADS_PER_CYCLE, 0, Config::LOAD_VEC_SIZE>::Invoke(
			key_ranks, key_digits, digit_prefixes);
	} 
	
};


// Recover digit counts for each load from composite 8-bit counters recorded in lane_totals
// Called by threads [0 .. radix-digits - 1]
template <typename Config> 
struct RecoverTileDigitCounts
{
	// Next load
	template <int CYCLE, int TOTAL_CYCLES, int LOAD, int TOTAL_LOADS>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
			int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],
			int my_base_lane,
			int my_quad_byte)
		{
			unsigned char *composite_counter = reinterpret_cast<unsigned char *>(
				&lane_totals[CYCLE][my_base_lane + (Config::SCAN_LANES_PER_LOAD * LOAD)]);

			digit_counts[CYCLE][LOAD] = composite_counter[my_quad_byte];

			// Correct for possible overflow
			if (WarpVoteAll(Config::RADIX_BITS, digit_counts[CYCLE][LOAD] <= 1)) {
				// We overflowed in this load: all keys for this load have same
				// digit, i.e., whoever owns the digit matching one of their own 
				// key's digit gets all 256 counts
				digit_counts[CYCLE][LOAD] = (threadIdx.x == key_digits[CYCLE][LOAD][0]) ? 256 : 0;
			}

			Iterate<CYCLE, TOTAL_CYCLES, LOAD + 1, TOTAL_LOADS>::Invoke(
				key_digits, lane_totals, digit_counts, my_base_lane, my_quad_byte);
		}
	};

	// Next cycle
	template <int CYCLE, int TOTAL_CYCLES, int TOTAL_LOADS>
	struct Iterate<CYCLE, TOTAL_CYCLES, TOTAL_LOADS, TOTAL_LOADS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
			int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],
			int my_base_lane,
			int my_quad_byte)
		{
			Iterate<CYCLE + 1, TOTAL_CYCLES, 0, TOTAL_LOADS>::Invoke(
				key_digits, lane_totals, digit_counts, my_base_lane, my_quad_byte);
		}
	};
	
	// Terminate
	template <int TOTAL_CYCLES, int TOTAL_LOADS>
	struct Iterate<TOTAL_CYCLES, TOTAL_CYCLES, 0, TOTAL_LOADS>
	{
		static __device__ __forceinline__ void Invoke(
			int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
			int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],
			int my_base_lane,
			int my_quad_byte) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		int key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
		int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE])				// Counts of my digit in each load in each cycle 
	{
		int my_base_lane = threadIdx.x >> 2;
		int my_quad_byte = threadIdx.x & 3;

		Iterate<0, Config::CYCLES_PER_TILE, 0, Config::LOADS_PER_CYCLE>::Invoke(
				key_digits, lane_totals, digit_counts, my_base_lane, my_quad_byte);
	} 
};


// Compute the digit prefixes for this tile for each load  
// Called by threads [0 .. radix-digits - 1]
template <typename Config> 
struct UpdateDigitPrefixes
{
	// Next load
	template <int CYCLE, int TOTAL_CYCLES, int LOAD, int TOTAL_LOADS>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			int digit_prefix,																
			int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],				
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS]) 
		{
			digit_prefixes[CYCLE][LOAD][threadIdx.x] = digit_counts[CYCLE][LOAD] + digit_prefix;
			Iterate<CYCLE, TOTAL_CYCLES, LOAD + 1, TOTAL_LOADS>::Invoke(
				digit_prefix, digit_counts, digit_prefixes);
		}
	};

	// Next cycle
	template <int CYCLE, int TOTAL_CYCLES, int TOTAL_LOADS>
	struct Iterate<CYCLE, TOTAL_CYCLES, TOTAL_LOADS, TOTAL_LOADS>
	{
		static __device__ __forceinline__ void Invoke(
			int digit_prefix,																
			int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],				
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS]) 
		{
			Iterate<CYCLE + 1, TOTAL_CYCLES, 0, TOTAL_LOADS>::Invoke(
				digit_prefix, digit_counts, digit_prefixes);
		}
	};
	
	// Terminate
	template <int TOTAL_CYCLES, int TOTAL_LOADS>
	struct Iterate<TOTAL_CYCLES, TOTAL_CYCLES, 0, TOTAL_LOADS>
	{
		static __device__ __forceinline__ void Invoke(
			int digit_prefix,																
			int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],				
			int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS]) {} 
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		int digit_prefix,																// My digit's prefix for this tile
		int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE],				// Counts of my digit in each load in each cycle of this tile
		int digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS])
	{
		Iterate<0, Config::CYCLES_PER_TILE, 0, Config::LOADS_PER_CYCLE>::Invoke(
			digit_prefix, digit_counts, digit_prefixes);
	} 
};


// Scans all of the cycles in a tile
template <typename Config, bool UNGUARDED_IO>
struct ScanTileCycles
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::SizeT SizeT;
	
	// Next cycle
	template <int CYCLE, int TOTAL_CYCLES>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			int 		*base_partial,
			int			*raking_segment,
			int 		lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE],
			KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE])
		{
			// Byte offset from base_partial of 8-bit counter for each key
			int counter_offsets[Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];		

			// Reset smem composite counters
			#pragma unroll
			for (int SCAN_LANE = 0; SCAN_LANE < Config::SCAN_LANES_PER_CYCLE; SCAN_LANE++) {
				base_partial[SCAN_LANE * Config::Grid::ROWS_PER_LANE * Config::Grid::PADDED_PARTIALS_PER_ROW] = 0;
			}

			// Decode digits and update 8-bit composite counters for the keys in this cycle
			DecodeCycleKeys<Config>::Invoke(
				keys[CYCLE], 
				key_digits[CYCLE], 
				counter_offsets, 
				base_partial); 

			__syncthreads();
			
			// Use our raking threads to, in aggregate, scan the composite counter lanes
			if (threadIdx.x < Config::Grid::RAKING_THREADS) {

				ScanCycleLanes<Config>(raking_segment, lanes_warpscan, lane_totals[CYCLE]);
			}
			
			__syncthreads();

			// Extract the local ranks of each key
			ExtractCycleRanks<Config>::Invoke(
				key_ranks[CYCLE], 
				key_digits[CYCLE],
				counter_offsets,
				base_partial); 

			// Next cycle
			Iterate<CYCLE + 1, TOTAL_CYCLES>::Invoke(
				base_partial, 
				raking_segment, 
				lanes_warpscan, 
				keys, 
				key_digits, 
				key_ranks, 
				lane_totals);
		}
	};
	
	// Terminate
	template <int TOTAL_CYCLES>
	struct Iterate<TOTAL_CYCLES, TOTAL_CYCLES>
	{
		static __device__ __forceinline__ void Invoke(
			int 		*base_partial,
			int			*raking_segment,
			int 		lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE],
			KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE]) {} 
	};
	
	// Interface
	static __device__ __forceinline__ void Invoke(
		int 		*base_partial,
		int			*raking_segment,
		int 		lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE],
		KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int 		lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE]) 
	{
		Iterate<0, Config::CYCLES_PER_TILE>::Invoke(
			base_partial, 
			raking_segment, 
			lanes_warpscan, 
			keys, 
			key_digits, 
			key_ranks, 
			lane_totals);
	}
	
};


template <typename Config>
struct ComputeScatterOffsets
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::SizeT SizeT;

	// Iterate over loads
	template <int LOAD, int TOTAL_LOADS>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(
			SizeT 	scatter_offsets[Config::TILE_ELEMENTS_PER_THREAD],
			SizeT	digit_carry[Config::RADIX_DIGITS],
			KeyType 	linear_keys[Config::TILE_ELEMENTS_PER_THREAD])
		{
			scatter_offsets[LOAD] = digit_carry[DecodeDigit<Config>(linear_keys[LOAD])] + (Config::THREADS * LOAD) + threadIdx.x;
			Iterate<LOAD + 1, TOTAL_LOADS>::Invoke(scatter_offsets, digit_carry, linear_keys);
		}
	};

	// Terminate
	template <int TOTAL_LOADS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS>
	{
		static __device__ __forceinline__ void Invoke(
			SizeT 	scatter_offsets[Config::TILE_ELEMENTS_PER_THREAD],
			SizeT	digit_carry[Config::RADIX_DIGITS],
			KeyType 	linear_keys[Config::TILE_ELEMENTS_PER_THREAD]) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		SizeT 	scatter_offsets[Config::TILE_ELEMENTS_PER_THREAD],
		SizeT	digit_carry[Config::RADIX_DIGITS],
		KeyType 	linear_keys[Config::TILE_ELEMENTS_PER_THREAD])
	{
		Iterate<0, Config::TILE_ELEMENTS_PER_THREAD>::Invoke(scatter_offsets, digit_carry, linear_keys);
	}
};


template <typename Config, bool UNGUARDED_IO, bool STRICT_ALIGNMENT>
struct SwapAndScatter
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::ValueType ValueType;
	typedef typename Config::SizeT SizeT;

	template <typename V, int __dummy = 0> 
	struct PermuteValues
	{
		static __device__ __forceinline__ void Invoke(
			V * __restrict d_in_values,
			V * __restrict d_out_values,
			const SizeT	&guarded_elements,
			int *exchange,
			int linear_ranks[Config::TILE_ELEMENTS_PER_THREAD],
			SizeT scatter_offsets[Config::TILE_ELEMENTS_PER_THREAD])
		{
			// Read values
			V values[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE];
			V *linear_values = reinterpret_cast<ValueType *>(values);
			V *value_exchange = reinterpret_cast<ValueType *>(exchange);

			LoadTile<
				V,													// Type to load
				SizeT,											// Integer type for indexing into global arrays
				Config::LOG_LOADS_PER_TILE, 						// Number of vector loads (log)
				Config::LOG_LOAD_VEC_SIZE,							// Number of items per vector load (log)
				Config::THREADS,									// Active threads that will be loading
				Config::CACHE_MODIFIER,								// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
				UNGUARDED_IO>										// Whether or not bounds-checking is to be done
			::Invoke(values, d_in_values, 0, guarded_elements);

			__syncthreads();

			// Scatter values to smem
			Scatter<ValueType, int, Config::TILE_ELEMENTS_PER_THREAD, Config::THREADS, NONE, true>::Invoke(
					value_exchange, linear_values, linear_ranks, guarded_elements);

			__syncthreads();

			// Gather values from smem (vec-1)
			LoadTile<ValueType, int, Config::LOG_TILE_ELEMENTS_PER_THREAD, 0, Config::THREADS, NONE, true>::Invoke(
				reinterpret_cast<ValueType (*)[1]>(values), value_exchange, 0, guarded_elements);

			// Scatter values to global digit partitions
			Scatter<ValueType, SizeT, Config::TILE_ELEMENTS_PER_THREAD, Config::THREADS, Config::CACHE_MODIFIER, UNGUARDED_IO>::Invoke(
				d_out_values, linear_values, scatter_offsets, guarded_elements);
		}
	};
	
	template <int __dummy> 
	struct PermuteValues <KeysOnly, __dummy>
	{
		static __device__ __forceinline__ void Invoke(
			KeysOnly * __restrict d_in_values,
			KeysOnly * __restrict d_out_values,
			const SizeT	&guarded_elements,
			int *exchange,
			int linear_ranks[Config::TILE_ELEMENTS_PER_THREAD],
			SizeT scatter_offsets[Config::TILE_ELEMENTS_PER_THREAD]) {}
	};

	
	static __device__ __forceinline__ void Invoke(
		ValueType 	* __restrict d_in_values,
		KeyType 	* __restrict d_out_keys,
		ValueType 	* __restrict d_out_values,
		int 		*exchange,
		SizeT	digit_carry[Config::RADIX_DIGITS],
		const SizeT	&guarded_elements,
		KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],			// The keys this thread will read this tile
		int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE])		// The CTA-scope rank of each key
	{
		KeyType *linear_keys = reinterpret_cast<KeyType *>(keys);
		KeyType *key_exchange = reinterpret_cast<KeyType *>(exchange);
		int *linear_ranks = reinterpret_cast<int *>(key_ranks);

		// Scatter keys to smem
		Scatter<KeyType, int, Config::TILE_ELEMENTS_PER_THREAD, Config::THREADS, NONE, true>::Invoke(
			key_exchange, linear_keys, linear_ranks, guarded_elements);

		__syncthreads();

		// Gather keys from smem (vec-1)
		LoadTile<KeyType, int, Config::LOG_TILE_ELEMENTS_PER_THREAD, 0, Config::THREADS, NONE, true>::Invoke(
			reinterpret_cast<KeyType (*)[1]>(keys), key_exchange, 0, guarded_elements);
		
		// Compute global scatter offsets for gathered keys
		SizeT scatter_offsets[Config::TILE_ELEMENTS_PER_THREAD];
		ComputeScatterOffsets<Config>::Invoke(scatter_offsets, digit_carry, linear_keys);

		// Scatter keys to global digit partitions
		Scatter<
			KeyType,
			SizeT,
			Config::TILE_ELEMENTS_PER_THREAD,
			Config::THREADS,
			Config::CACHE_MODIFIER,
			UNGUARDED_IO,
			Config::PostprocessTraits::Postprocess>::Invoke(d_out_keys, linear_keys, scatter_offsets, guarded_elements);

		// PermuteValues
		PermuteValues<ValueType>::Invoke(
			d_in_values,
			d_out_values,
			guarded_elements,
			exchange,
			linear_ranks, 
			scatter_offsets);
		
	}
};





/******************************************************************************
 * SM1.0 Local Exchange Routines
 *
 * Routines for exchanging keys (and values) in shared memory (i.e., local 
 * scattering) in order to to facilitate coalesced global scattering
 ******************************************************************************/

/*
template <
	typename T, 
	int RADIX_DIGITS,
	bool UNGUARDED_IO,
	typename PostprocessTraits> 
__device__ __forceinline__ void ScatterCycle(
	T *swapmem,
	T *d_out, 
	int digit_warpscan[2][RADIX_DIGITS], 
	int digit_carry[RADIX_DIGITS], 
	const int &partial_tile_elements,
	int base_digit,				
	PostprocessTraits postprocess = PostprocessTraits())				
{
	const int LOG_STORE_TXN_THREADS = B40C_LOG_MEM_BANKS(__CUDA_ARCH__);
	const int STORE_TXN_THREADS = 1 << LOG_STORE_TXN_THREADS;
	
	int store_txn_idx = threadIdx.x & (STORE_TXN_THREADS - 1);
	int store_txn_digit = threadIdx.x >> LOG_STORE_TXN_THREADS;
	
	int my_digit = base_digit + store_txn_digit;
	if (my_digit < RADIX_DIGITS) {
	
		int my_exclusive_scan = digit_warpscan[1][my_digit - 1];
		int my_inclusive_scan = digit_warpscan[1][my_digit];
		int my_digit_count = my_inclusive_scan - my_exclusive_scan;

		int my_carry = digit_carry[my_digit] + my_exclusive_scan;
		int my_aligned_offset = store_txn_idx - (my_carry & (STORE_TXN_THREADS - 1));
		
		while (my_aligned_offset < my_digit_count) {

			if ((my_aligned_offset >= 0) && (UNGUARDED_IO || (my_exclusive_scan + my_aligned_offset < partial_tile_elements))) { 
			
				T datum = swapmem[my_exclusive_scan + my_aligned_offset];
				postprocess(datum);
				d_out[my_carry + my_aligned_offset] = datum;
			}
			my_aligned_offset += STORE_TXN_THREADS;
		}
	}
}

template <
	typename T,
	int RADIX_DIGITS, 
	int CYCLES_PER_TILE,
	int LOADS_PER_CYCLE,
	bool UNGUARDED_IO,
	typename PostprocessTraits>
__device__ __forceinline__ void SwapAndScatterPairs(
	typename VecType<T, 2>::Type pairs[CYCLES_PER_TILE][LOADS_PER_CYCLE], 
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	T *exchange,
	T *d_out, 
	int digit_carry[RADIX_DIGITS], 
	int digit_warpscan[2][RADIX_DIGITS], 
	const int &partial_tile_elements)				
{
	const int SCATTER_CYCLE_DIGITS = B40C_RADIXSORT_WARPS * (B40C_WARP_THREADS / B40C_MEM_BANKS(__CUDA_ARCH__));
	const int SCATTER_CYCLES = RADIX_DIGITS / SCATTER_CYCLE_DIGITS;

	// Push in pairs
	PushPairs<T, CYCLES_PER_TILE, LOADS_PER_CYCLE>(exchange, pairs, ranks);

	__syncthreads();

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, not an innermost loop"

	if (SCATTER_CYCLES > 0) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 0);
	if (SCATTER_CYCLES > 1) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 1);
	if (SCATTER_CYCLES > 2) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 2);
	if (SCATTER_CYCLES > 3) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 3);
	if (SCATTER_CYCLES > 4) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 4);
	if (SCATTER_CYCLES > 5) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 5);
	if (SCATTER_CYCLES > 6) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 6);
	if (SCATTER_CYCLES > 7) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessTraits>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 7);
}


template <
	typename KeyType,
	typename ValueType,	
	CacheModifier CACHE_MODIFIER,
	int RADIX_DIGITS, 
	int CYCLES_PER_TILE,
	int LOADS_PER_CYCLE,
	bool UNGUARDED_IO,
	typename PostprocessTraits>
__device__ __forceinline__ void SwapAndScatterSm10(
	typename VecType<KeyType, 2>::Type keypairs[CYCLES_PER_TILE][LOADS_PER_CYCLE], 
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int *exchange,
	typename VecType<ValueType, 2>::Type *d_in_values, 
	KeyType *d_out_keys, 
	ValueType *d_out_values, 
	int digit_carry[RADIX_DIGITS], 
	int digit_warpscan[2][RADIX_DIGITS], 
	const int &partial_tile_elements)				
{
	// Swap and scatter keys
	SwapAndScatterPairs<KeyType, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessTraits>(
		keypairs, ranks, (KeyType*) exchange, d_out_keys, digit_carry, digit_warpscan, partial_tile_elements);				
	
	if (!IsKeysOnly<ValueType>()) {

		__syncthreads();
		
		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		// Read input data
		typename VecType<ValueType, 2>::Type datapairs[CYCLES_PER_TILE][LOADS_PER_CYCLE];
		if (CYCLES_PER_TILE > 0) ReadCycle<ValueType, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<ValueType> >::Read(d_in_values, datapairs[0], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 0, partial_tile_elements);
		if (CYCLES_PER_TILE > 1) ReadCycle<ValueType, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<ValueType> >::Read(d_in_values, datapairs[1], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 1, partial_tile_elements);

		// Swap and scatter data
		SwapAndScatterPairs<ValueType, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, NopFunctor<ValueType> >(
			datapairs, ranks, (ValueType*) exchange, d_out_values, digit_carry, digit_warpscan, partial_tile_elements);				
	}
}
*/


template <
	typename Config,
	bool UNGUARDED_IO>
__device__ __forceinline__ void ProcessTile(
	typename Config::KeyType 	* __restrict d_in_keys, 
	typename Config::ValueType 	* __restrict d_in_values, 
	typename Config::KeyType 	* __restrict d_out_keys, 
	typename Config::ValueType 	* __restrict d_out_values, 
	int 						*exchange,								
	int							lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE],
	typename Config::SizeT		digit_carry[Config::RADIX_DIGITS],
	int							digit_warpscan[2][Config::RADIX_DIGITS],						 
	int							digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS],
	int 						lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
	int							*base_partial,
	int							*raking_segment,		
	const typename Config::SizeT 		&guarded_elements)
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::SizeT SizeT;
	
	KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];			// The keys this thread will read this tile
	int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];	// Their decoded digits
	int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];		// The CTA-scope rank of each key

	// Read tile of keys
	LoadTile<
		KeyType,											// Type to load
		SizeT,											// Integer type for indexing into global arrays
		Config::LOG_LOADS_PER_TILE, 						// Number of vector loads (log)
		Config::LOG_LOAD_VEC_SIZE,							// Number of items per vector load (log)
		Config::THREADS,									// Active threads that will be loading
		Config::CACHE_MODIFIER,								// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
		UNGUARDED_IO,										// Whether or not bounds-checking is to be done
		Config::PreprocessTraits::Preprocess>				// Assignment function to transform the loaded value (or provide default if out-of-bounds)
	::Invoke(
			reinterpret_cast<KeyType (*)[Config::LOAD_VEC_SIZE]>(keys),	 
			d_in_keys,
			0,
			guarded_elements);

	// Scan cycles
	ScanTileCycles<Config, UNGUARDED_IO>::Invoke(
		base_partial,
		raking_segment,
		lanes_warpscan,
		keys,
		key_digits,
		key_ranks,
		lane_totals); 

	// Scan across digits
	if (threadIdx.x < Config::RADIX_DIGITS) {
		int digit_counts[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE];					// Counts of my digit in each load in each cycle
		
		// Recover digit-counts from lanes_warpscan padding
		RecoverTileDigitCounts<Config>::Invoke(key_digits, lane_totals, digit_counts);
		
		// Scan across my digit counts for each load
		int inclusive_total = SerialScan<int, Config::LOADS_PER_TILE>::Invoke(
			reinterpret_cast<int*>(digit_counts), 0);

		// Add the inclusive scan of digit counts from the previous tile to the running carry
		SizeT my_carry = digit_carry[threadIdx.x] + digit_warpscan[1][threadIdx.x];

		// Perform overflow-free SIMD Kogge-Stone across digits
		int digit_prefix_inclusive = WarpScanInclusive<int, Config::RADIX_BITS>::Invoke(
				inclusive_total,
				digit_warpscan);

		// Save inclusive scan in digit_warpscan
		digit_warpscan[1][threadIdx.x] = digit_prefix_inclusive;

		// Calculate exclusive scan
		int digit_prefix_exclusive = digit_prefix_inclusive - inclusive_total;

		// Subtract the digit prefix from the running carry (to offset threadIdx during scatter)
		digit_carry[threadIdx.x] = my_carry - digit_prefix_exclusive;

		// Compute the digit prefixes for this tile for each load  
		UpdateDigitPrefixes<Config>::Invoke(digit_prefix_exclusive, digit_counts, digit_prefixes);
	}
	
	__syncthreads();

	// Update the key ranks in each load with the digit prefixes for the tile
	UpdateTileRanks<Config>::Invoke(key_ranks, key_digits, digit_prefixes);

	// Scatter to outgoing digit partitions (using a local scatter first)
	SwapAndScatter<Config, UNGUARDED_IO, (__B40C_CUDA_ARCH__ < 120)>::Invoke(
		d_in_values,
		d_out_keys,
		d_out_values,
		exchange,
		digit_carry,
		guarded_elements,
		keys,
		key_ranks);

	__syncthreads();
}


template <typename Config>
__device__ __forceinline__ void DigitPass(
	typename Config::SizeT 		* __restrict d_spine,
	typename Config::KeyType 	* __restrict d_in_keys, 
	typename Config::ValueType 	* __restrict d_in_values, 
	typename Config::KeyType 	* __restrict d_out_keys, 
	typename Config::ValueType 	* __restrict d_out_values, 
	int 						*exchange,								
	int							lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE],
	typename Config::SizeT		digit_carry[Config::RADIX_DIGITS],
	int							digit_warpscan[2][Config::RADIX_DIGITS],						 
	int							digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS],
	int 						lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
	int							*base_partial,
	int							*raking_segment,		
	typename Config::SizeT 		cta_offset,
	typename Config::SizeT 		&guarded_offset,
	typename Config::SizeT 		&guarded_elements)
{
	typedef typename Config::SizeT SizeT;

	if (threadIdx.x < Config::RADIX_DIGITS) {

		// Reset value-area of digit_warpscan
		digit_warpscan[1][threadIdx.x] = 0;

		// Read digit_carry in parallel 
		SizeT my_digit_carry;
		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		ModifiedLoad<SizeT, Config::CACHE_MODIFIER>::Ld(my_digit_carry, d_spine, spine_digit_offset);
		digit_carry[threadIdx.x] = my_digit_carry;
	}

	// Scan in full tiles of tile_elements
	while (cta_offset < guarded_offset) {

		ProcessTile<Config, true>(
			d_in_keys + cta_offset,
			d_in_values + cta_offset,
			d_out_keys,
			d_out_values,
			exchange,
			lanes_warpscan,
			digit_carry,
			digit_warpscan,						 
			digit_prefixes,
			lane_totals,
			base_partial,
			raking_segment,
			0);
	
		cta_offset += Config::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (guarded_elements) {

		ProcessTile<Config, false>(	
			d_in_keys + cta_offset,
			d_in_values + cta_offset,
			d_out_keys,
			d_out_values,
			exchange,
			lanes_warpscan,
			digit_carry,
			digit_warpscan,						 
			digit_prefixes,
			lane_totals,
			base_partial,
			raking_segment,
			guarded_elements);
	}

}



/**
 * Downsweep scan-scatter 
 */
template <typename Config>
__device__ __forceinline__ void LsbDownsweep(
	int 						* __restrict &d_selectors,
	typename Config::SizeT 		* __restrict &d_spine,
	typename Config::KeyType 	* __restrict &d_keys0,
	typename Config::KeyType 	* __restrict &d_keys1,
	typename Config::ValueType 	* __restrict &d_values0,
	typename Config::ValueType 	* __restrict &d_values1,
	CtaWorkDistribution<typename Config::SizeT> &work_decomposition)
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::SizeT SizeT;

	__shared__ int4			smem_pool_int4s[Config::SMEM_POOL_INT4S];
	__shared__ int 			lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::Grid::RAKING_THREADS_PER_LANE];		// One warpscan per lane
	__shared__ SizeT	digit_carry[Config::RADIX_DIGITS];
	__shared__ int 			digit_warpscan[2][Config::RADIX_DIGITS];
	__shared__ int 			digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS];
	__shared__ int 			lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE];
	__shared__ bool 		non_trivial_digit_pass;
	__shared__ int 			selector;
	__shared__ SizeT 	cta_offset;			// Offset at which this CTA begins processing
	__shared__ SizeT 	guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	__shared__ SizeT 	guarded_elements;		// Number of elements in partially-full tile
	
	int *smem_pool = reinterpret_cast<int *>(smem_pool_int4s);

	// location for placing 2-element partial reductions in the first lane of a cycle	
	int *base_partial = Config::Grid::BasePartial(smem_pool); 								
	
	// location for raking across all loads within a cycle
	int *raking_segment = 0;										
	
	if (threadIdx.x < Config::Grid::RAKING_THREADS) {

		// initalize lane warpscans
		int warpscan_lane = threadIdx.x >> Config::Grid::LOG_RAKING_THREADS_PER_LANE;
		int warpscan_tid = threadIdx.x & (Config::Grid::RAKING_THREADS_PER_LANE - 1);
		lanes_warpscan[warpscan_lane][0][warpscan_tid] = 0;
		
		// initialize raking segment
		raking_segment = Config::Grid::RakingSegment(smem_pool); 

		// initialize digit warpscans
		if (threadIdx.x < Config::RADIX_DIGITS) {

			// Initialize digit_warpscan
			digit_warpscan[0][threadIdx.x] = 0;

			// Determine where to read our input
			if (Config::EARLY_EXIT) {

				// We have early-exit-upon-homogeneous-digits enabled

				const int SELECTOR_IDX = Config::CURRENT_PASS & 0x1;
				const int NEXT_SELECTOR_IDX = (Config::CURRENT_PASS + 1) & 0x1;

				selector = (Config::CURRENT_PASS == 0) ? 0 : d_selectors[SELECTOR_IDX];

				// Determine whether or not we have work to do and setup the next round
				// accordingly.  We can do this by looking at the first-block's
				// histograms and counting the number of digits with counts that are
				// non-zero and not-the-problem-size.
				if (Config::PreprocessTraits::MustApply || Config::PostprocessTraits::MustApply) {
					non_trivial_digit_pass = true;
				} else {
					int first_block_carry = d_spine[FastMul(gridDim.x, threadIdx.x)];
					int predicate = ((first_block_carry > 0) && (first_block_carry < work_decomposition.num_elements));
					non_trivial_digit_pass = TallyWarpVote(Config::RADIX_BITS, predicate, smem_pool);
				}

				// Let the next round know which set of buffers to use
				if (blockIdx.x == 0) {
					d_selectors[NEXT_SELECTOR_IDX] = selector ^ non_trivial_digit_pass;
				}
			}

			// Determine our threadblock's work range
			SizeT cta_elements;			// Total number of elements for this CTA to process
			work_decomposition.GetCtaWorkLimits<Config::LOG_TILE_ELEMENTS, Config::LOG_SCHEDULE_GRANULARITY>(
				cta_offset, cta_elements, guarded_offset, guarded_elements);
		}
	}

	// Sync to acquire non_trivial_digit_pass and selector
	__syncthreads();
	
	// Short-circuit this entire cycle
	if (Config::EARLY_EXIT && !non_trivial_digit_pass) return;

	if ((Config::EARLY_EXIT && selector) || (!Config::EARLY_EXIT && (Config::CURRENT_PASS & 0x1))) {
	
		// d_keys1 -> d_keys0
		DigitPass<Config>(	
			d_spine,
			d_keys1,
			d_values1,
			d_keys0, 
			d_values0, 
			smem_pool,
			lanes_warpscan,
			digit_carry,
			digit_warpscan,						 
			digit_prefixes,
			lane_totals,
			base_partial,
			raking_segment,
			cta_offset,
			guarded_offset,
			guarded_elements);

	} else {
		
		// d_keys0 -> d_keys1
		DigitPass<Config>(	
			d_spine,
			d_keys0,
			d_values0,
			d_keys1, 
			d_values1, 
			smem_pool,
			lanes_warpscan,
			digit_carry,
			digit_warpscan,						 
			digit_prefixes,
			lane_totals,
			base_partial,
			raking_segment,
			cta_offset,
			guarded_offset,
			guarded_elements);
	}
}


/**
 * Downsweep scan-scatter kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void DownsweepKernel(
	int 								* __restrict d_selectors,
	typename KernelConfig::SizeT 		* __restrict d_spine,
	typename KernelConfig::KeyType 		* __restrict d_keys0,
	typename KernelConfig::KeyType 		* __restrict d_keys1,
	typename KernelConfig::ValueType 	* __restrict d_values0,
	typename KernelConfig::ValueType 	* __restrict d_values1,
	CtaWorkDistribution<typename KernelConfig::SizeT> work_decomposition)
{
	LsbDownsweep<KernelConfig>(
		d_selectors,
		d_spine,
		d_keys0,
		d_keys1,
		d_values0,
		d_values1,
		work_decomposition);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename KernelConfig>
void __wrapper__device_stub_DownsweepKernel(
	int 								* __restrict &,
	typename KernelConfig::SizeT 		* __restrict &,
	typename KernelConfig::KeyType 		* __restrict &,
	typename KernelConfig::KeyType 		* __restrict &,
	typename KernelConfig::ValueType 	* __restrict &,
	typename KernelConfig::ValueType 	* __restrict &,
	CtaWorkDistribution<typename KernelConfig::SizeT> &) {}



} // namespace downsweep
} // namespace radix_sort
} // namespace b40c

