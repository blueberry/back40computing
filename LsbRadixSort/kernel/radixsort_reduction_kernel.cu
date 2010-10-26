/******************************************************************************
 * 
 * Copyright 2010 Duane Merrill
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
 * 
 * 
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
 * Bottom-level digit-reduction/counting kernel
 ******************************************************************************/

#pragma once

#include "radixsort_kernel_common.cu"

namespace b40c {


/******************************************************************************
 * Defines
 ******************************************************************************/

const int BYTE_ENCODE_SHIFT = 0x3;


/******************************************************************************
 * Cycle-processing Routines
 ******************************************************************************/

__device__ __forceinline__ int DecodeInt(int encoded, int quad_byte){
	return (encoded >> quad_byte) & 0xff;		// shift right 8 bits per digit and return rightmost 8 bits
}


__device__ __forceinline__ int EncodeInt(int count, int quad_byte) {
	return count << quad_byte;					// shift left 8 bits per digit
}


template <typename K, long long RADIX_DIGITS, int BIT>
__device__ __forceinline__ void DecodeDigit(
	K key, 
	int &lane, 
	int &quad_shift) 
{
	const K DIGIT_MASK = RADIX_DIGITS - 1;
	lane = (key & (DIGIT_MASK << BIT)) >> (BIT + 2);
	
	const K QUAD_MASK = (RADIX_DIGITS < 4) ? 0x1 : 0x3;
	if (BIT == 32) {
		// N.B.: This takes one more instruction than the code below it, but 
		// otherwise the compiler goes nuts and shoves hundreds of bytes 
		// to lmem when bit = 32 on 64-bit keys.		
		quad_shift = ((key >> BIT) & QUAD_MASK) << BYTE_ENCODE_SHIFT;	
	} else {
		quad_shift = MagnitudeShift<K, BYTE_ENCODE_SHIFT - BIT>(key & (QUAD_MASK << BIT));
	}
}


template <
	int REDUCTION_LANES, 
	int REDUCTION_LANES_PER_WARP, 
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE, 
	bool FINAL_REDUCE>
__device__ __forceinline__ void ReduceEncodedCounts(
	int local_counts[REDUCTION_LANES_PER_WARP][4],
	int *smem) 
{
	const int LANE_PARTIALS_PER_THREAD 	= REDUCTION_PARTIALS_PER_LANE / B40C_WARP_THREADS;
	
	int idx = threadIdx.x & (B40C_WARP_THREADS - 1);
	int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS;
	int lane_base = warp_id << LOG_REDUCTION_PARTIALS_PER_LANE;	// my warp's (first) reduction lane

	__syncthreads();
	
	#pragma unroll
	for (int j = 0; j < (int) REDUCTION_LANES_PER_WARP; j++) {
		
		if (lane_base < REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE) {

			// rest of my elements
			#pragma unroll
			for (int i = 0; i < (int) LANE_PARTIALS_PER_THREAD; i++) {
				int encoded = smem[lane_base + idx + (i * B40C_WARP_THREADS)];		
				local_counts[j][0] += DecodeInt(encoded, 0u << BYTE_ENCODE_SHIFT);
				local_counts[j][1] += DecodeInt(encoded, 1u << BYTE_ENCODE_SHIFT);
				local_counts[j][2] += DecodeInt(encoded, 2u << BYTE_ENCODE_SHIFT);
				local_counts[j][3] += DecodeInt(encoded, 3u << BYTE_ENCODE_SHIFT);
			}
			
			if (FINAL_REDUCE) {
				// reduce all four packed fields, leaving them in the first four elements of our row
				WarpReduce<B40C_WARP_THREADS>(idx, &smem[lane_base + 0], local_counts[j][0]);
				WarpReduce<B40C_WARP_THREADS>(idx, &smem[lane_base + 1], local_counts[j][1]);
				WarpReduce<B40C_WARP_THREADS>(idx, &smem[lane_base + 2], local_counts[j][2]);
				WarpReduce<B40C_WARP_THREADS>(idx, &smem[lane_base + 3], local_counts[j][3]);
			}
			
			lane_base += REDUCTION_PARTIALS_PER_LANE * B40C_RADIXSORT_WARPS;
		}
	}	

	__syncthreads();
	
}
	

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
__device__ __forceinline__ void Bucket(
	K input, 
	int *encoded_reduction_col,
	PreprocessFunctor preprocess = PreprocessFunctor()) 
{
	int lane, quad_shift;
	preprocess(input);
	DecodeDigit<K, RADIX_DIGITS, BIT>(input, lane, quad_shift);
	encoded_reduction_col[FastMul(lane, REDUCTION_PARTIALS_PER_LANE)] += EncodeInt(1, quad_shift);
}


template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor, int CYCLES>
struct LoadOp;

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		K key = d_in_keys[reduction_offset + threadIdx.x];
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(key, encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 1), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 2), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
			K keys[8];
				
			keys[0] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 0) + threadIdx.x];
			keys[1] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 1) + threadIdx.x];
			keys[2] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 2) + threadIdx.x];
			keys[3] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 3) + threadIdx.x];

			if (B40C_FERMI(__CUDA_ARCH__)) __syncthreads();
			
			keys[4] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 4) + threadIdx.x];
			keys[5] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 5) + threadIdx.x];
			keys[6] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 6) + threadIdx.x];
			keys[7] = d_in_keys[reduction_offset + (B40C_RADIXSORT_THREADS * 7) + threadIdx.x];
			
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[0], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[1], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[2], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[3], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[4], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[5], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[6], encoded_reduction_col);
			Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[7], encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 8), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 16), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 32), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 128> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 64), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 252> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int reduction_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 128>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 128), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 192), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 224), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 240), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, reduction_offset + (B40C_RADIXSORT_THREADS * 248), encoded_reduction_col);
	}
};


template <int REDUCTION_LANES>
__device__ __forceinline__ void ResetEncodedCarry(
	int *encoded_reduction_col)
{
	#pragma unroll
	for (int SCAN_LANE = 0; SCAN_LANE < (int) REDUCTION_LANES; SCAN_LANE++) {
		encoded_reduction_col[SCAN_LANE * B40C_RADIXSORT_THREADS] = 0;
	}
}


template <
	typename K, 
	int RADIX_DIGITS, 
	int REDUCTION_LANES, 
	int REDUCTION_LANES_PER_WARP, 
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE, 
	int BIT, 
	typename PreprocessFunctor>
__device__ __forceinline__ int ProcessLoads(
	K *d_in_keys,
	int loads,
	int &reduction_offset,
	int *encoded_reduction_col,
	int *smem,
	int local_counts[REDUCTION_LANES_PER_WARP][4])
{
	// Unroll batches of loads with occasional reduction to avoid overflow
	while (loads >= 128) {
	
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 128>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 128;
		loads -= 128;

		// Reduce int local count registers to prevent overflow
		ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, false>(
				local_counts, 
				smem);
		
		// Reset encoded counters
		ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);
	} 
	
	int retval = loads;
	
	// Wind down loads in decreasing batch sizes
	if (loads >= 64) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 64;
		loads -= 64;
	}
	if (loads >= 32) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 32;
		loads -= 32;
	}
	if (loads >= 16) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 16;
		loads -= 16;
	}
	if (loads >= 8) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 8;
		loads -= 8;
	}
	if (loads >= 4) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 4;
		loads -= 4;
	}
	if (loads >= 2) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 2;
		loads -= 2;
	}
	if (loads) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, reduction_offset, encoded_reduction_col);
		reduction_offset += B40C_RADIXSORT_THREADS * 1;
	}
	
	return retval;
}


/******************************************************************************
 * Reduction/counting Kernel Entry Point
 ******************************************************************************/

template <typename K, typename V, int PASS, int RADIX_BITS, int BIT, typename PreprocessFunctor>
__launch_bounds__ (B40C_RADIXSORT_THREADS, B40C_RADIXSORT_REDUCE_CTA_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void RakingReduction(
	int *d_selectors,
	int *d_spine,
	K *d_in_keys,
	K *d_out_keys,
	CtaDecomposition work_decomposition)
{
	const int RADIX_DIGITS 						= 1 << RADIX_BITS;

	const int LOG_REDUCTION_PARTIALS_PER_LANE	= B40C_RADIXSORT_LOG_THREADS;
	const int REDUCTION_PARTIALS_PER_LANE 		= 1 << LOG_REDUCTION_PARTIALS_PER_LANE;

	const int LOG_REDUCTION_LANES 				= (RADIX_BITS >= 2) ? RADIX_BITS - 2 : 0;	// Always at least one fours group
	const int REDUCTION_LANES 					= 1 << LOG_REDUCTION_LANES;

	const int LOG_REDUCTION_LANES_PER_WARP 		= (REDUCTION_LANES > B40C_RADIXSORT_WARPS) ? LOG_REDUCTION_LANES - B40C_RADIXSORT_LOG_WARPS : 0;	// Always at least one fours group per warp
	const int REDUCTION_LANES_PER_WARP 			= 1 << LOG_REDUCTION_LANES_PER_WARP;
	
	const int REDUCTION_LOADS_PER_CYCLE 		= B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V) / B40C_RADIXSORT_THREADS;
	
	
	
	// Each thread gets its own column of fours-groups (for conflict-free updates)
	__shared__ int smem[REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE];	
	
	int *encoded_reduction_col = &smem[threadIdx.x];	// first element of column

	// Each thread is also responsible for aggregating an unencoded segment of a fours-group
	int local_counts[REDUCTION_LANES_PER_WARP][4];								

	// Determine where to read our input
	int selector = (PASS == 0) ? 0 : d_selectors[PASS & 0x1];
	if (selector) d_in_keys = d_out_keys;
	
	// Calculate our threadblock's range
	int reduction_offset, block_elements;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		reduction_offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		reduction_offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * B40C_RADIXSORT_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V));
		block_elements = work_decomposition.normal_block_elements;
	}
	
	// Initialize local counts
	#pragma unroll 
	for (int LANE = 0; LANE < (int) REDUCTION_LANES_PER_WARP; LANE++) {
		local_counts[LANE][0] = 0;
		local_counts[LANE][1] = 0;
		local_counts[LANE][2] = 0;
		local_counts[LANE][3] = 0;
	}
	
	// Reset encoded counters
	ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);
	
	// Process loads
	int loads = block_elements >> B40C_RADIXSORT_LOG_THREADS;
	int unreduced_loads = ProcessLoads<K, RADIX_DIGITS, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(
		d_in_keys,
		loads,
		reduction_offset,
		encoded_reduction_col,
		smem,
		local_counts);

	// Cleanup if we're the last block  
	if ((blockIdx.x == gridDim.x - 1) && (work_decomposition.extra_elements_last_block)) {

		// If extra guarded loads may cause overflow, reduce now and reset counters
		if (unreduced_loads + REDUCTION_LOADS_PER_CYCLE > 255) {
		
			ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, false>(
					local_counts, 
					smem);
			
			ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);
		}

		// perform up to REDUCTION_LOADS_PER_CYCLE extra guarded loads
		#pragma unroll
		for (int EXTRA_LOAD = 0; EXTRA_LOAD < (int) REDUCTION_LOADS_PER_CYCLE; EXTRA_LOAD++) {

			const int LOAD_BASE = B40C_RADIXSORT_THREADS * EXTRA_LOAD;
			
			if (LOAD_BASE + threadIdx.x < work_decomposition.extra_elements_last_block) {

				K key = d_in_keys[reduction_offset + LOAD_BASE + threadIdx.x];
				Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(key, encoded_reduction_col);
				
			}
		}
	}

	// Aggregate 
	ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, true>(
		local_counts, 
		smem);

	// Write carry in parallel (carries per row are in the first four bytes of each row) 
	if (threadIdx.x < RADIX_DIGITS) {
		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		int row = threadIdx.x >> 2;		
		int col = threadIdx.x & 3;			 

		d_spine[spine_digit_offset] = smem[(row * B40C_RADIXSORT_THREADS) + col];
	}
} 

 

} // namespace b40c

