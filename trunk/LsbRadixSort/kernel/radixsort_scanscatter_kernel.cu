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
 * Radix sorting downsweep scan-scatter kernel.  The third kernel in a radix-
 * sorting digit-place pass.
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.cu"

namespace b40c {
namespace lsb_radix_sort {
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
	typename _IndexType,
	int _RADIX_BITS,
	int _LOG_SUBTILE_ELEMENTS,
	int _CTA_OCCUPANCY,
	int _LOG_THREADS,
	int _LOG_LOAD_VEC_SIZE,
	int _LOG_LOADS_PER_CYCLE,
	int _LOG_CYCLES_PER_TILE,
	int _LOG_RAKING_THREADS,
	CacheModifier _CACHE_MODIFIER>

struct DownsweepConfig
{
	typedef _KeyType							KeyType;
	typedef _ValueType							ValueType;
	typedef _IndexType							IndexType;
	static const int RADIX_BITS					= _RADIX_BITS;
	static const int LOG_SUBTILE_ELEMENTS		= _LOG_SUBTILE_ELEMENTS;
	static const int CTA_OCCUPANCY  			= _CTA_OCCUPANCY;
	static const int LOG_THREADS 				= _LOG_THREADS;
	static const int LOG_LOAD_VEC_SIZE 			= _LOG_LOAD_VEC_SIZE;
	static const int LOG_LOADS_PER_CYCLE		= _LOG_LOADS_PER_CYCLE;
	static const int LOG_CYCLES_PER_TILE		= _LOG_CYCLES_PER_TILE;
	static const int LOG_RAKING_THREADS			= _LOG_RAKING_THREADS;
	static const CacheModifier CACHE_MODIFIER 	= _CACHE_MODIFIER;
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
	typename 		PreprocessFunctorType, 
	typename 		PostprocessFunctorType, 
	int 			_CURRENT_PASS,
	int 			_CURRENT_BIT>
struct DownsweepKernelConfig : DownsweepConfigType
{
	typedef PreprocessFunctorType					PreprocessFunctor;
	typedef PostprocessFunctorType					PostprocessFunctor;

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
	
	static const int LOG_SCAN_LANES_PER_LOAD		= B40C_MAX((DownsweepConfigType::RADIX_BITS - 2), 0);		// Always at one lane per load
	static const int SCAN_LANES_PER_LOAD			= 1 << LOG_SCAN_LANES_PER_LOAD;								
	
	static const int LOG_SCAN_LANES_PER_CYCLE		= DownsweepConfigType::LOG_LOADS_PER_CYCLE + LOG_SCAN_LANES_PER_LOAD;
	static const int SCAN_LANES_PER_CYCLE			= 1 << LOG_SCAN_LANES_PER_CYCLE;
	
	static const int LOG_PARTIALS_PER_LANE 			= DownsweepConfigType::LOG_THREADS;
	
	static const int LOG_PARTIALS_PER_CYCLE			= LOG_SCAN_LANES_PER_CYCLE + LOG_PARTIALS_PER_LANE;

	static const int LOG_RAKING_THREADS_PER_LANE 	= DownsweepConfigType::LOG_RAKING_THREADS - LOG_SCAN_LANES_PER_CYCLE;
	static const int RAKING_THREADS_PER_LANE 		= 1 << LOG_RAKING_THREADS_PER_LANE;

	// Smem SRTS grid type for reducing and scanning a cycle of 
	// (radix-digits/4) lanes of composite 8-bit digit counters
	typedef SrtsGrid<
		int,											// type
		DownsweepConfigType::LOG_THREADS,				// depositing threads
		LOG_SCAN_LANES_PER_CYCLE, 						// deposits per thread
		DownsweepConfigType::LOG_RAKING_THREADS> 		// raking threads
			Grid;
	
	static const int LOG_ROWS_PER_LOAD 				= LOG_PARTIALS_PER_CYCLE - Grid::LOG_PARTIALS_PER_ROW;

	static const int LOG_ROWS_PER_LANE 				= LOG_PARTIALS_PER_LANE - Grid::LOG_PARTIALS_PER_ROW;
	static const int ROWS_PER_LANE 					= 1 << LOG_ROWS_PER_LANE;

	static const int LOG_ROWS_PER_CYCLE 			= LOG_SCAN_LANES_PER_CYCLE + LOG_ROWS_PER_LANE;
	static const int ROWS_PER_CYCLE 				= 1 << LOG_ROWS_PER_CYCLE;

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
			const int PADDED_BYTES_PER_LANE 	= Config::ROWS_PER_LANE * Config::Grid::PADDED_PARTIALS_PER_ROW * 4;
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
	int 	lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::RAKING_THREADS_PER_LANE],
	int 	lane_totals[Config::SCAN_LANES_PER_CYCLE])
{
	// Upsweep rake
	int partial = SerialReduce<int, Config::Grid::PARTIALS_PER_SEG>::Invoke(raking_segment);

	// Warpscan reduction in digit warpscan_lane
	int warpscan_total;
	int warpscan_lane = threadIdx.x >> Config::LOG_RAKING_THREADS_PER_LANE;
	int warpscan_tid = threadIdx.x & (Config::RAKING_THREADS_PER_LANE - 1);

	int prefix = WarpScan<int, Config::RAKING_THREADS_PER_LANE>::Invoke(
		partial, warpscan_total, lanes_warpscan[warpscan_lane], warpscan_tid);
	
	// Save off each lane's warpscan total for this cycle 
	lane_totals[warpscan_lane] = warpscan_total;

	// Downsweep rake
	SerialScan<int, Config::Grid::PARTIALS_PER_SEG>::Invoke(raking_segment, prefix);
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
			if (WarpVoteAll(Config::RADIX_DIGITS, digit_counts[CYCLE][LOAD] <= 1)) {
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
	typedef typename Config::IndexType IndexType;
	
	// Next cycle
	template <int CYCLE, int TOTAL_CYCLES>
	struct Iterate 
	{
		static __device__ __forceinline__ void Invoke(
			KeyType		*d_in_keys,
			IndexType	cta_offset,
			IndexType	cta_out_of_bounds,
			int 		*base_partial,
			int			*raking_segment,
			int 		lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::RAKING_THREADS_PER_LANE],
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
				base_partial[SCAN_LANE * Config::ROWS_PER_LANE * Config::Grid::PADDED_PARTIALS_PER_ROW ] = 0;
			}
			
			// Read cycle of keys
			LoadTile<
				KeyType,											// Type to load
				IndexType,											// Integer type for indexing into global arrays 
				Config::LOG_LOADS_PER_CYCLE, 						// Number of vector loads (log)
				Config::LOG_LOAD_VEC_SIZE,							// Number of items per vector load (log)
				Config::THREADS,									// Active threads that will be loading
				Config::CACHE_MODIFIER,								// Cache modifier (e.g., CA/CG/CS/NONE/etc.)
				UNGUARDED_IO,										// Whether or not bounds-checking is to be done
				Config::PreprocessFunctor::Transform>				// Assignment function to transform the loaded value (or provide default if out-of-bounds)
			::Invoke(
					keys[CYCLE],	 
					d_in_keys,
					cta_offset + (CYCLE * Config::CYCLE_ELEMENTS),
					cta_out_of_bounds);
			
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
				d_in_keys, 
				cta_offset, 
				cta_out_of_bounds, 
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
			KeyType		*d_in_keys,
			IndexType	cta_offset,
			IndexType	cta_out_of_bounds,
			int 		*base_partial,
			int			*raking_segment,
			int 		lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::RAKING_THREADS_PER_LANE],
			KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
			int 		lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE]) {} 
	};
	
	// Interface
	static __device__ __forceinline__ void Invoke(
		KeyType		*d_in_keys,
		IndexType	cta_offset,
		IndexType	cta_out_of_bounds,
		int 		*base_partial,
		int			*raking_segment,
		int 		lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::RAKING_THREADS_PER_LANE],
		KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],
		int 		lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE]) 
	{
		Iterate<0, Config::CYCLES_PER_TILE>::Invoke(
			d_in_keys, 
			cta_offset, 
			cta_out_of_bounds, 
			base_partial, 
			raking_segment, 
			lanes_warpscan, 
			keys, 
			key_digits, 
			key_ranks, 
			lane_totals);
	}
	
};



/******************************************************************************
 * SM1.3 Local Exchange Routines
 * 
 * Routines for exchanging keys (and values) in shared memory (i.e., local 
 * scattering) in order to to facilitate coalesced global scattering
 ******************************************************************************/

/*
template <typename T, bool UNGUARDED_IO, int CYCLES_PER_TILE, int LOADS_PER_CYCLE, typename PostprocessFunctor>
__device__ __forceinline__ void ScatterLoads(
	T *d_out, 
	typename VecType<T, 2>::Type pairs[LOADS_PER_CYCLE],
	int2 offsets[LOADS_PER_CYCLE],
	const int BASE4,
	const int &partial_tile_elements,
	PostprocessFunctor postprocess = PostprocessFunctor())				
{
	#pragma unroll 
	for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
		postprocess(pairs[LOAD].x);
		postprocess(pairs[LOAD].y);

		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * (LOAD * 2 + 0)) < partial_tile_elements)) 
			d_out[offsets[LOAD].x] = pairs[LOAD].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (B40C_RADIXSORT_THREADS * (LOAD * 2 + 1)) < partial_tile_elements)) 
			d_out[offsets[LOAD].y] = pairs[LOAD].y;
	}
}

template <typename T, int CYCLES_PER_TILE, int LOADS_PER_CYCLE>
__device__ __forceinline__ void PushPairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE])				
{
	#pragma unroll 
	for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
	
		#pragma unroll 
		for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
			swap[ranks[CYCLE][LOAD].x] = pairs[CYCLE][LOAD].x;
			swap[ranks[CYCLE][LOAD].y] = pairs[CYCLE][LOAD].y;
		}
	}
}
	
template <typename T, int CYCLES_PER_TILE, int LOADS_PER_CYCLE>
__device__ __forceinline__ void ExchangePairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[CYCLES_PER_TILE][LOADS_PER_CYCLE],
	int2 ranks[CYCLES_PER_TILE][LOADS_PER_CYCLE])				
{
	// Push in Pairs
	PushPairs<T, CYCLES_PER_TILE, LOADS_PER_CYCLE>(swap, pairs, ranks);
	
	__syncthreads();
	
	// Extract pairs
	#pragma unroll 
	for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
		
		#pragma unroll 
		for (int LOAD = 0; LOAD < (int) LOADS_PER_CYCLE; LOAD++) {
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			pairs[CYCLE][LOAD].x = swap[threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0))];
			pairs[CYCLE][LOAD].y = swap[threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1))];
		}
	}
}
*/

template <
	typename Config, 
	bool UNGUARDED_IO>
__device__ __forceinline__ void SwapAndScatterSm13(
	typename Config::KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],			// The keys this thread will read this tile
	int 						key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE],			// The CTA-scope rank of each key
	int 						*exchange,
	typename Config::IndexType 	digit_carry[Config::RADIX_DIGITS],
	int 						partial_tile_elements,
	typename Config::KeyType 	*d_out_keys, 
	typename Config::ValueType 	*d_in_values, 
	typename Config::ValueType 	*d_out_values)				
{
	
	
	
/*	
	int2 offsets[CYCLES_PER_TILE][LOADS_PER_CYCLE];
	
	// Swap keys according to ranks
	ExchangePairs<KeyType, CYCLES_PER_TILE, LOADS_PER_CYCLE>((KeyType*) exchange, keypairs, ranks);				
	
	// Calculate scatter offsets (re-decode digits from keys: it's less work than making a second exchange of digits)
	if (CYCLES_PER_TILE > 0) {
		const int CYCLE = 0;
		if (LOADS_PER_CYCLE > 0) {
			const int LOAD = 0;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
		if (LOADS_PER_CYCLE > 1) {
			const int LOAD = 1;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
	}
	if (CYCLES_PER_TILE > 1) {
		const int CYCLE = 1;
		if (LOADS_PER_CYCLE > 0) {
			const int LOAD = 0;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
		if (LOADS_PER_CYCLE > 1) {
			const int LOAD = 1;
			const int BLOCK = ((CYCLE * LOADS_PER_CYCLE) + LOAD) * 2;
			offsets[CYCLE][LOAD].x = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 0)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].x)];
			offsets[CYCLE][LOAD].y = threadIdx.x + (B40C_RADIXSORT_THREADS * (BLOCK + 1)) + digit_carry[DecodeDigit<KeyType, RADIX_BITS, BIT>(keypairs[CYCLE][LOAD].y)];
		}
	}
	
	// Scatter keys
	#pragma unroll 
	for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
		const int BLOCK = CYCLE * LOADS_PER_CYCLE * 2;
		ScatterLoads<KeyType, UNGUARDED_IO, CYCLES_PER_TILE, LOADS_PER_CYCLE, PostprocessFunctor>(d_out_keys, keypairs[CYCLE], offsets[CYCLE], B40C_RADIXSORT_THREADS * BLOCK, partial_tile_elements);
	}

	if (!IsKeysOnly<ValueType>()) {
	
		__syncthreads();

		// Read input data
		typename VecType<ValueType, 2>::Type datapairs[CYCLES_PER_TILE][LOADS_PER_CYCLE];

		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		if (CYCLES_PER_TILE > 0) ReadCycle<ValueType, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<ValueType> >::Read(d_in_values, datapairs[0], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 0, partial_tile_elements);
		if (CYCLES_PER_TILE > 1) ReadCycle<ValueType, CACHE_MODIFIER, UNGUARDED_IO, LOADS_PER_CYCLE, NopFunctor<ValueType> >::Read(d_in_values, datapairs[1], B40C_RADIXSORT_THREADS * LOADS_PER_CYCLE * 1, partial_tile_elements);
		
		// Swap data according to ranks
		ExchangePairs<ValueType, CYCLES_PER_TILE, LOADS_PER_CYCLE>((ValueType*) exchange, datapairs, ranks);
		
		// Scatter data
		#pragma unroll 
		for (int CYCLE = 0; CYCLE < (int) CYCLES_PER_TILE; CYCLE++) {
			const int BLOCK = CYCLE * LOADS_PER_CYCLE * 2;
			ScatterLoads<ValueType, UNGUARDED_IO, CYCLES_PER_TILE, LOADS_PER_CYCLE, NopFunctor<ValueType> >(d_out_values, datapairs[CYCLE], offsets[CYCLE], B40C_RADIXSORT_THREADS * BLOCK, partial_tile_elements);
		}
	}
*/	
}


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
	typename PostprocessFunctor> 
__device__ __forceinline__ void ScatterCycle(
	T *swapmem,
	T *d_out, 
	int digit_warpscan[2][RADIX_DIGITS], 
	int digit_carry[RADIX_DIGITS], 
	const int &partial_tile_elements,
	int base_digit,				
	PostprocessFunctor postprocess = PostprocessFunctor())				
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
	typename PostprocessFunctor>
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

	if (SCATTER_CYCLES > 0) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 0);
	if (SCATTER_CYCLES > 1) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 1);
	if (SCATTER_CYCLES > 2) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 2);
	if (SCATTER_CYCLES > 3) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 3);
	if (SCATTER_CYCLES > 4) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 4);
	if (SCATTER_CYCLES > 5) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 5);
	if (SCATTER_CYCLES > 6) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 6);
	if (SCATTER_CYCLES > 7) ScatterCycle<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_warpscan, digit_carry, partial_tile_elements, SCATTER_CYCLE_DIGITS * 7);
}


template <
	typename KeyType,
	typename ValueType,	
	CacheModifier CACHE_MODIFIER,
	int RADIX_DIGITS, 
	int CYCLES_PER_TILE,
	int LOADS_PER_CYCLE,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
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
	SwapAndScatterPairs<KeyType, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessFunctor>(
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
	typename Config::KeyType 	*d_in_keys, 
	typename Config::ValueType 	*d_in_values, 
	typename Config::KeyType 	*d_out_keys, 
	typename Config::ValueType 	*d_out_values, 
	int 						*exchange,								
	int							lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::RAKING_THREADS_PER_LANE],
	typename Config::IndexType	digit_carry[Config::RADIX_DIGITS],
	int							digit_warpscan[2][Config::RADIX_DIGITS],						 
	int							digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS],
	int 						lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
	int							*base_partial,
	int							*raking_segment,		
	typename Config::IndexType 	cta_offset,
	typename Config::IndexType 	cta_out_of_bounds)
{
	typedef typename Config::KeyType KeyType;
	typedef typename Config::IndexType IndexType;
	
	KeyType 	keys[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];			// The keys this thread will read this tile
	int 		key_digits[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];	// Their decoded digits
	int 		key_ranks[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::LOAD_VEC_SIZE];		// The CTA-scope rank of each key

	// Scan cycles
	ScanTileCycles<Config, UNGUARDED_IO>::Invoke(
		d_in_keys,
		cta_offset,
		cta_out_of_bounds,
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

		printf("Block %d digit %d has count %d\n", blockIdx.x, threadIdx.x, inclusive_total);


		// Second half of digit_carry update
		IndexType my_carry = digit_carry[threadIdx.x] + digit_warpscan[1][threadIdx.x];
		
		// Perform overflow-free SIMD Kogge-Stone across digits
		int tile_total;
		int digit_prefix = WarpScan<int, Config::RADIX_DIGITS>::Invoke(
				inclusive_total,
				tile_total,
				digit_warpscan); 

		// first-half of digit_carry update 
		digit_carry[threadIdx.x] = my_carry - digit_prefix;

		// Compute the digit prefixes for this tile for each load  
		UpdateDigitPrefixes<Config>::Invoke(digit_prefix, digit_counts, digit_prefixes);
	}
	
	__syncthreads();

	// Update the key ranks in each load with the digit prefixes for the tile
	UpdateTileRanks<Config>::Invoke(key_ranks, key_digits, digit_prefixes);
	
	
/*	
	
	//-------------------------------------------------------------------------
	// Scatter 
	//-------------------------------------------------------------------------

#if ((__CUDA_ARCH__ < 130) || FERMI_ECC)		

	SwapAndScatterSm10<KeyType, ValueType, CACHE_MODIFIER, RADIX_DIGITS, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		digit_carry, 
		digit_warpscan,
		partial_tile_elements);
	
#else 

	SwapAndScatterSm13<KeyType, ValueType, CACHE_MODIFIER, RADIX_BITS, RADIX_DIGITS, BIT, CYCLES_PER_TILE, LOADS_PER_CYCLE, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_values, 
		d_out_keys, 
		d_out_values, 
		digit_carry, 
		partial_tile_elements);
	
#endif
*/
	
	__syncthreads();
	
}


template <typename Config>
__device__ __forceinline__ void DigitPass(
	typename Config::IndexType 	*d_spine,
	typename Config::KeyType 	*d_in_keys, 
	typename Config::ValueType 	*d_in_values, 
	typename Config::KeyType 	*d_out_keys, 
	typename Config::ValueType 	*d_out_values, 
	int 						*exchange,								
	int							lanes_warpscan[Config::SCAN_LANES_PER_CYCLE][3][Config::RAKING_THREADS_PER_LANE],
	int							digit_carry[Config::RADIX_DIGITS],
	int							digit_warpscan[2][Config::RADIX_DIGITS],						 
	int							digit_prefixes[Config::CYCLES_PER_TILE][Config::LOADS_PER_CYCLE][Config::RADIX_DIGITS],
	int 						lane_totals[Config::CYCLES_PER_TILE][Config::SCAN_LANES_PER_CYCLE],
	int							*base_partial,
	int							*raking_segment,		
	typename Config::IndexType 	cta_offset,
	typename Config::IndexType 	guarded_offset,
	typename Config::IndexType 	cta_out_of_bounds)
{
	typedef typename Config::IndexType IndexType;
	
	if (threadIdx.x < Config::RADIX_DIGITS) {

		// Reset value-area of digit_warpscan
		digit_warpscan[1][threadIdx.x] = 0;

		// Read digit_carry in parallel 
		IndexType my_digit_carry;
		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		ModifiedLoad<IndexType, Config::CACHE_MODIFIER>::Ld(my_digit_carry, d_spine, spine_digit_offset);
		digit_carry[threadIdx.x] = my_digit_carry;
	}
	
/*
	// Scan in full tiles of tile_elements
	while (cta_offset < guarded_offset) {

		ProcessTile<Config, true>(	
			d_in_keys,
			d_in_values,
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
			cta_offset,
			cta_out_of_bounds);
	
		cta_offset += Config::TILE_ELEMENTS;
	}
*/
	
	// Clean up last partial tile with guarded-io
	if (cta_offset < cta_out_of_bounds) {

		ProcessTile<Config, false>(	
			d_in_keys,
			d_in_values,
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
			cta_offset,
			cta_out_of_bounds);
	}
}



/**
 * Downsweep scan-scatter kernel entry point
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void LsbScanScatterKernel(
	int 								*d_selectors,
	typename KernelConfig::IndexType 	*d_spine,
	typename KernelConfig::KeyType 		*d_keys0,
	typename KernelConfig::KeyType 		*d_keys1,
	typename KernelConfig::ValueType 	*d_values0,
	typename KernelConfig::ValueType 	*d_values1,
	CtaDecomposition<typename KernelConfig::IndexType> work_decomposition)
{
	typedef typename KernelConfig::KeyType KeyType;
	typedef typename KernelConfig::IndexType IndexType;

	__shared__ int4		smem_pool_int4s[KernelConfig::SMEM_POOL_INT4S];
	__shared__ int 		lanes_warpscan[KernelConfig::SCAN_LANES_PER_CYCLE][3][KernelConfig::RAKING_THREADS_PER_LANE];		// One warpscan per lane
	__shared__ int 		digit_carry[KernelConfig::RADIX_DIGITS];
	__shared__ int 		digit_warpscan[2][KernelConfig::RADIX_DIGITS];						 
	__shared__ int 		digit_prefixes[KernelConfig::CYCLES_PER_TILE][KernelConfig::LOADS_PER_CYCLE][KernelConfig::RADIX_DIGITS];
	__shared__ int 		lane_totals[KernelConfig::CYCLES_PER_TILE][KernelConfig::SCAN_LANES_PER_CYCLE];
	__shared__ bool 	non_trivial_digit_pass;
	__shared__ int 		selector;
	
	int *smem_pool = reinterpret_cast<int *>(smem_pool_int4s);
	
	
	// Determine our threadblock's work range
	
	IndexType cta_offset;			// Offset at which this CTA begins processing
	IndexType cta_elements;			// Total number of elements for this CTA to process
	IndexType guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	IndexType guarded_elements;		// Number of elements in partially-full tile 
	IndexType cta_out_of_bounds;

	work_decomposition.GetCtaWorkLimits<KernelConfig::LOG_TILE_ELEMENTS, KernelConfig::LOG_SUBTILE_ELEMENTS>(
		cta_offset, cta_elements, guarded_offset, guarded_elements);
	cta_out_of_bounds = cta_offset + cta_elements;
	
	// location for placing 2-element partial reductions in the first lane of a cycle	
	int *base_partial = KernelConfig::Grid::BasePartial(smem_pool); 								
	
	// location for raking across all loads within a cycle
	int *raking_segment = 0;										
	
	if (threadIdx.x < KernelConfig::Grid::RAKING_THREADS) {

		// initalize lane warpscans
		int warpscan_lane = threadIdx.x >> KernelConfig::LOG_RAKING_THREADS_PER_LANE;
		int warpscan_tid = threadIdx.x & (KernelConfig::RAKING_THREADS_PER_LANE - 1);
		lanes_warpscan[warpscan_lane][0][warpscan_tid] = 0;
		
		// initialize raking segment
		raking_segment = KernelConfig::Grid::RakingSegment(smem_pool); 
	}

	// initialize digit warpscans
	if (threadIdx.x < KernelConfig::RADIX_DIGITS) {

		const int SELECTOR_IDX = KernelConfig::CURRENT_PASS & 0x1;
		const int NEXT_SELECTOR_IDX = (KernelConfig::CURRENT_PASS + 1) & 0x1;
		
		// Initialize digit_warpscan
		digit_warpscan[0][threadIdx.x] = 0;

		// Determine where to read our input
		selector = (KernelConfig::CURRENT_PASS == 0) ? 0 : d_selectors[SELECTOR_IDX];

		// Determine whether or not we have work to do and setup the next round 
		// accordingly.  We can do this by looking at the first-block's 
		// histograms and counting the number of digits with counts that are 
		// non-zero and not-the-problem-size.
		if (KernelConfig::PreprocessFunctor::MustApply || KernelConfig::PostprocessFunctor::MustApply) {
			non_trivial_digit_pass = true;
		} else {
			int first_block_carry = d_spine[FastMul(gridDim.x, threadIdx.x)];
			int predicate = ((first_block_carry > 0) && (first_block_carry < work_decomposition.num_elements));
			non_trivial_digit_pass = TallyWarpVote(KernelConfig::RADIX_DIGITS, predicate, smem_pool);
		}

		// Let the next round know which set of buffers to use
		if (blockIdx.x == 0) {
			d_selectors[NEXT_SELECTOR_IDX] = selector ^ non_trivial_digit_pass;
		}
	}

	// Sync to acquire non_trivial_digit_pass and selector
	__syncthreads();
	
	// Short-circuit this entire cycle
	if (!non_trivial_digit_pass) return;
	
//	if (!selector) {
	
		// d_keys0 -> d_keys1 

		DigitPass<KernelConfig>(	
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
			cta_out_of_bounds);		
/*

	} else {
		
		// d_keys1 -> d_keys0
		DigitPass<KernelConfig>(	
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
			cta_out_of_bounds);		
	}
*/
}




/**
 * Host stub to calm the linker for arch-specializations that we didn't
 * end up compiling PTX for.
 */
template <typename KernelConfig>
__host__ void __wrapper__device_stub_LsbScanScatterKernel(
	int 								*&,
	typename KernelConfig::IndexType 	*&,
	typename KernelConfig::KeyType 		*&,
	typename KernelConfig::KeyType 		*&,
	typename KernelConfig::ValueType 	*&,
	typename KernelConfig::ValueType 	*&,
	CtaDecomposition<typename KernelConfig::IndexType> &) {}



} // namespace downsweep
} // namespace lsb_radix_sort
} // namespace b40c

