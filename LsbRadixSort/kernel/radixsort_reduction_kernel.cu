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

template <int BYTE>
__device__ __forceinline__ int DecodeInt(int encoded){
	
	int retval;
	ExtractBits<int, BYTE * 8, 8>(retval, encoded);
	return retval;
}


template <typename K, long long RADIX_DIGITS, int BIT>
__device__ __forceinline__ void DecodeDigit(
	K key, 
	int &lane, 
	int &quad_byte) 
{
/*	
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
*/	

	const int RADIX_BITS = 4;

	ExtractBits<K, BIT + 2, RADIX_BITS - 2>(lane, key);
	if (RADIX_BITS < 2) { 
		ExtractBits<K, BIT, 1>(quad_byte, key);
	} else {
		ExtractBits<K, BIT, 2>(quad_byte, key);
	}
}

//-----------------------------------------------------------------------------

template <int PARTIAL>
__device__ __forceinline__  void ReduceLanePartial(
	int local_counts[4], 
	int *scan_lanes, 
	int lane_offset) 
{
	unsigned char* encoded = (unsigned char *) &scan_lanes[lane_offset + (PARTIAL * B40C_WARP_THREADS)];
	local_counts[0] += encoded[0];
	local_counts[1] += encoded[1];
	local_counts[2] += encoded[2];
	local_counts[3] += encoded[3];
/*	
	int encoded = scan_lanes[lane_offset + (PARTIAL * B40C_WARP_THREADS)];		
	local_counts[0] += DecodeInt<0>(encoded);
	local_counts[1] += DecodeInt<1>(encoded);
	local_counts[2] += DecodeInt<2>(encoded);
	local_counts[3] += DecodeInt<3>(encoded);
*/	
}

template <int LANE, int REDUCTION_LANES, int REDUCTION_LANES_PER_WARP, int REDUCTION_PARTIALS_PER_LANE, int LANE_PARTIALS_PER_THREAD>
__device__ __forceinline__  void ReduceLanePartials(
	int local_counts[REDUCTION_LANES_PER_WARP][4],
	int *scan_lanes, 
	int lane_offset) 
{		
	lane_offset += (LANE * REDUCTION_PARTIALS_PER_LANE * B40C_RADIXSORT_WARPS);
	if (lane_offset < REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE) {
		if (LANE_PARTIALS_PER_THREAD > 0) ReduceLanePartial<0>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 1) ReduceLanePartial<1>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 2) ReduceLanePartial<2>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 3) ReduceLanePartial<3>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 4) ReduceLanePartial<4>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 5) ReduceLanePartial<5>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 6) ReduceLanePartial<6>(local_counts[LANE], scan_lanes, lane_offset);
		if (LANE_PARTIALS_PER_THREAD > 7) ReduceLanePartial<7>(local_counts[LANE], scan_lanes, lane_offset);
	}
}


template <
	int REDUCTION_LANES, 
	int REDUCTION_LANES_PER_WARP, 
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE> 
__device__ __forceinline__ void ReduceEncodedCounts(
	int local_counts[REDUCTION_LANES_PER_WARP][4],
	int *scan_lanes,
	int warp_id,
	int warp_idx)
{
	const int LANE_PARTIALS_PER_THREAD = REDUCTION_PARTIALS_PER_LANE / B40C_WARP_THREADS;
	SuppressUnusedConstantWarning(LANE_PARTIALS_PER_THREAD);
	
	int lane_offset = (warp_id << LOG_REDUCTION_PARTIALS_PER_LANE) + warp_idx;	// my warp's (first-lane) reduction offset

	if (REDUCTION_LANES_PER_WARP > 0) ReduceLanePartials<0, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
	if (REDUCTION_LANES_PER_WARP > 1) ReduceLanePartials<1, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
	if (REDUCTION_LANES_PER_WARP > 2) ReduceLanePartials<2, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
	if (REDUCTION_LANES_PER_WARP > 3) ReduceLanePartials<3, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, REDUCTION_PARTIALS_PER_LANE, LANE_PARTIALS_PER_THREAD>(local_counts, scan_lanes, lane_offset);
}
	

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
__device__ __forceinline__ void Bucket(
	K input, 
	int *encoded_reduction_col,
	PreprocessFunctor preprocess = PreprocessFunctor()) 
{
	int lane, quad_byte;
	preprocess(input);
	DecodeDigit<K, RADIX_DIGITS, BIT>(input, lane, quad_byte);
	
	unsigned char *encoded_col = (unsigned char *) &encoded_reduction_col[FastMul(lane, REDUCTION_PARTIALS_PER_LANE)];
	encoded_col[quad_byte]++;
}


template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor, int CYCLES>
struct LoadOp;

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		K key;
		LoadCop(key, &d_in_keys[block_offset]);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(key, encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 1), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 2>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 2), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8> 
{
	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		if (B40C_FERMI(__CUDA_ARCH__)) __syncthreads();
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 4>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 4), encoded_reduction_col);
		
/*		
		K keys[8];
			
		keys[0] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 0) + threadIdx.x];
		keys[1] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 1) + threadIdx.x];
		keys[2] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 2) + threadIdx.x];
		keys[3] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 3) + threadIdx.x];

		if (B40C_FERMI(__CUDA_ARCH__)) __syncthreads();
		
		keys[4] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 4) + threadIdx.x];
		keys[5] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 5) + threadIdx.x];
		keys[6] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 6) + threadIdx.x];
		keys[7] = d_in_keys[block_offset + (B40C_RADIXSORT_THREADS * 7) + threadIdx.x];
		
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[0], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[1], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[2], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[3], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[4], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[5], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[6], encoded_reduction_col);
		Bucket<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(keys[7], encoded_reduction_col);
*/		
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 8>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 8), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 16), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 32>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 32), encoded_reduction_col);
	}
};

template <typename K, int RADIX_DIGITS, int REDUCTION_PARTIALS_PER_LANE, int BIT, typename PreprocessFunctor>
struct LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 128> {

	static __device__ __forceinline__  void BlockOfLoads(K *d_in_keys, int block_offset, int *encoded_reduction_col)
	{
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 0), encoded_reduction_col);
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 64>::BlockOfLoads(d_in_keys, block_offset + (B40C_RADIXSORT_THREADS * 64), encoded_reduction_col);
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
__device__ __forceinline__ void ProcessLoads(
	K *d_in_keys,
	int loads,
	int &block_offset,
	int *encoded_reduction_col,
	int *scan_lanes,
	int local_counts[REDUCTION_LANES_PER_WARP][4],
	int warp_id,
	int warp_idx)
{
	// Unroll batches of loads with occasional reduction to avoid overflow
	while (loads >= 16) {
	
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 16>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
		block_offset += B40C_RADIXSORT_THREADS * 16;
		loads -= 16;

		__syncthreads();

		// Reduce int local count registers to prevent overflow
		ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE>(
				local_counts, 
				scan_lanes,
				warp_id,
				warp_idx);

		__syncthreads();
		
		// Reset encoded counters
		ResetEncodedCarry<REDUCTION_LANES>(encoded_reduction_col);
	} 
	while (loads) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
		block_offset += B40C_RADIXSORT_THREADS * 1;
		loads--;
	}
	
}


template <
	typename K, 
	typename V,
	int RADIX_DIGITS, 
	int REDUCTION_LANES, 
	int LOG_REDUCTION_PARTIALS_PER_LANE,
	int REDUCTION_PARTIALS_PER_LANE, 
	int BIT, 
	int TILE_ELEMENTS,
	typename PreprocessFunctor>
__device__ __forceinline__ void ReductionPass(
	K*			d_in_keys,
	int* 		d_spine,
	int 		block_offset,
	int 		reduction_loads,
	int* 		encoded_reduction_col,
	int*		scan_lanes,
	const int&	extra_elements)
{
	const int REDUCTION_LANES_PER_WARP 			= (REDUCTION_LANES > B40C_RADIXSORT_WARPS) ? REDUCTION_LANES / B40C_RADIXSORT_WARPS : 1;	// Always at least one fours group per warp
	const int PARTIALS_PER_ROW = B40C_WARP_THREADS;
	const int PADDED_PARTIALS_PER_ROW = PARTIALS_PER_ROW + 1;

	int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS;
	int warp_idx = threadIdx.x & (B40C_WARP_THREADS - 1);
	
	block_offset += threadIdx.x;
	
	// Each thread is responsible for aggregating an unencoded segment of a fours-group
	int local_counts[REDUCTION_LANES_PER_WARP][4];								
	
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
	ProcessLoads<K, RADIX_DIGITS, REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor>(
		d_in_keys,
		reduction_loads,
		block_offset,
		encoded_reduction_col,
		scan_lanes,
		local_counts, 
		warp_id,
		warp_idx);

	// Cleanup if we're the last block  
	if (threadIdx.x < extra_elements) {
		LoadOp<K, RADIX_DIGITS, REDUCTION_PARTIALS_PER_LANE, BIT, PreprocessFunctor, 1>::BlockOfLoads(d_in_keys, block_offset, encoded_reduction_col);
	}

	__syncthreads();

	// Aggregate 
	ReduceEncodedCounts<REDUCTION_LANES, REDUCTION_LANES_PER_WARP, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE>(
		local_counts, 
		scan_lanes,
		warp_id,
		warp_idx);
	
	__syncthreads();
		
	// reduce all four packed fields, leaving them in the first four elements of our row
	
	int lane_base = FastMul(warp_id, PADDED_PARTIALS_PER_ROW * B40C_RADIXSORT_WARPS);	// my warp's (first) reduction lane
	
	#pragma unroll
	for (int i = 0; i < (int) REDUCTION_LANES_PER_WARP; i++) {

		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 0)] = local_counts[i][0];
		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 1)] = local_counts[i][1];
		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 2)] = local_counts[i][2];
		scan_lanes[lane_base + warp_idx + (PADDED_PARTIALS_PER_ROW * 3)] = local_counts[i][3];
		
		lane_base += PADDED_PARTIALS_PER_ROW * B40C_RADIXSORT_WARPS;
	}

	__syncthreads();

	// Write carry in parallel (carries per row are in the first four bytes of each row) 
	if (threadIdx.x < RADIX_DIGITS) {

		int lane_base = FastMul(threadIdx.x, PADDED_PARTIALS_PER_ROW);
		int digit_count = SerialReduce<PARTIALS_PER_ROW>(scan_lanes + lane_base);

		int spine_digit_offset = FastMul(gridDim.x, threadIdx.x) + blockIdx.x;
		d_spine[spine_digit_offset] = digit_count;
	}

}




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
	const int TILE_ELEMENTS						= B40C_RADIXSORT_TILE_ELEMENTS(__CUDA_ARCH__, K, V);

	const int LOG_REDUCTION_PARTIALS_PER_LANE	= B40C_RADIXSORT_LOG_THREADS;
	const int REDUCTION_PARTIALS_PER_LANE 		= 1 << LOG_REDUCTION_PARTIALS_PER_LANE;

	const int LOG_REDUCTION_LANES 				= (RADIX_BITS >= 2) ? RADIX_BITS - 2 : 0;	// Always at least one fours group
	const int REDUCTION_LANES 					= 1 << LOG_REDUCTION_LANES;

	SuppressUnusedConstantWarning(RADIX_DIGITS);
	
	
	// Each thread gets its own column of fours-groups (for conflict-free updates)
	__shared__ int scan_lanes[REDUCTION_LANES * REDUCTION_PARTIALS_PER_LANE];	
	
	int *encoded_reduction_col = &scan_lanes[threadIdx.x];	// first element of column

	// Determine where to read our input
	int selector = (PASS == 0) ? 0 : d_selectors[PASS & 0x1];
	if (selector) d_in_keys = d_out_keys;
	
	// Calculate our threadblock's range
	int block_offset, block_elements;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		block_offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		block_offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * TILE_ELEMENTS);
		block_elements = work_decomposition.normal_block_elements;
	}
	int extra_elements, reduction_loads;
	if (blockIdx.x == gridDim.x - 1) {
		reduction_loads = (block_elements + work_decomposition.extra_elements_last_block) >> B40C_RADIXSORT_LOG_THREADS;
		extra_elements = work_decomposition.extra_elements_last_block & (B40C_RADIXSORT_THREADS - 1);
	} else {
		reduction_loads = block_elements >> B40C_RADIXSORT_LOG_THREADS;
		extra_elements = 0;
	}
	
	// Perform reduction pass
	ReductionPass<K, V, RADIX_DIGITS, REDUCTION_LANES, LOG_REDUCTION_PARTIALS_PER_LANE, REDUCTION_PARTIALS_PER_LANE, BIT, TILE_ELEMENTS, PreprocessFunctor>(
		d_in_keys,
		d_spine,
		block_offset,
		reduction_loads,
		encoded_reduction_col,
		scan_lanes,
		extra_elements);
} 

 

} // namespace b40c

