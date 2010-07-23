/**
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
 */


//------------------------------------------------------------------------------
// SrtsScanDigit
//------------------------------------------------------------------------------

#ifndef _SRTS_RADIX_SORT_SCANSCATTER_KERNEL_H_
#define _SRTS_RADIX_SORT_SCANSCATTER_KERNEL_H_

#include <kernel/srts_radixsort_kernel_common.cu>



//------------------------------------------------------------------------------
// Scan/Scatter Configuration
//------------------------------------------------------------------------------

#if ((__CUDA_ARCH__ >= 200) || (CUDA_VERSION < 3010))
	#define REG_MISER_QUALIFIER __shared__
#else
	#define REG_MISER_QUALIFIER 
#endif



//------------------------------------------------------------------------------
// Appropriate substitutes to use for out-of-bounds key (and value) offsets 
//------------------------------------------------------------------------------

template <typename T> 
__device__ __forceinline__ T DefaultextraValue() {
	return T();
}

template <> 
__device__ __forceinline__ unsigned char DefaultextraValue<unsigned char>() {
	return (unsigned char) -1;
}

template <> 
__device__ __forceinline__ unsigned short DefaultextraValue<unsigned short>() {
	return (unsigned short) -1;
}

template <> 
__device__ __forceinline__ unsigned int DefaultextraValue<unsigned int>() {
	return (unsigned int) -1;
}

template <> 
__device__ __forceinline__ unsigned long DefaultextraValue<unsigned long>() {
	return (unsigned long) -1;
}

template <> 
__device__ __forceinline__ unsigned long long DefaultextraValue<unsigned long long>() {
	return (unsigned long long) -1;
}


//------------------------------------------------------------------------------
// Cycle-processing Routines
//------------------------------------------------------------------------------

template <typename K, unsigned long long RADIX_DIGITS, unsigned int BIT>
__device__ __forceinline__ unsigned int DecodeDigit(K key) 
{
	const K DIGIT_MASK = RADIX_DIGITS - 1;
	return (key >> BIT) & DIGIT_MASK;
}


template <typename K, unsigned long long RADIX_DIGITS, unsigned int BIT, unsigned int PADDED_PARTIALS_PER_LANE>
__device__ __forceinline__ void DecodeDigit(
	K key, 
	unsigned int &digit, 
	unsigned int &flag_offset,		// in bytes
	const unsigned int SET_OFFSET)
{
	const unsigned int PADDED_BYTES_PER_LANE 	= PADDED_PARTIALS_PER_LANE * 4;
	const unsigned int SET_OFFSET_BYTES 		= SET_OFFSET * 4;
	const K QUAD_MASK 							= (RADIX_DIGITS < 4) ? 0x1 : 0x3;
	
	digit = DecodeDigit<K, RADIX_DIGITS, BIT>(key);
	unsigned int lane = digit >> 2;
	unsigned int quad_byte = digit & QUAD_MASK;

	flag_offset = SET_OFFSET_BYTES + FastMul(lane, PADDED_BYTES_PER_LANE) + quad_byte;
}


template <typename K, unsigned long long RADIX_DIGITS, unsigned int BIT, unsigned int SETS_PER_PASS, unsigned int SCAN_LANES_PER_SET, unsigned int PADDED_PARTIALS_PER_LANE>
__device__ __forceinline__ void DecodeDigits(
	typename VecType<K, 2>::Type keypairs[SETS_PER_PASS],
	uint2 digits[SETS_PER_PASS],
	uint2 flag_offsets[SETS_PER_PASS])		// in bytes 
{

	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		
		const unsigned int SET_OFFSET = SET * SCAN_LANES_PER_SET * PADDED_PARTIALS_PER_LANE;

		DecodeDigit<K, RADIX_DIGITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[SET].x, digits[SET].x, flag_offsets[SET].x, SET_OFFSET);
		
		DecodeDigit<K, RADIX_DIGITS, BIT, PADDED_PARTIALS_PER_LANE>(
				keypairs[SET].y, digits[SET].y, flag_offsets[SET].y, SET_OFFSET);
	}
}


template <typename T>
__device__ __forceinline__ void GuardedReadSet(
	T *in, 
	typename VecType<T, 2>::Type &pair,
	int offset,
	int extra[1])				
{
	pair.x = (offset - extra[0] < 0) ? in[offset] : DefaultextraValue<T>();
	pair.y = (offset + 1 - extra[0] < 0) ? in[offset + 1] : DefaultextraValue<T>();
}


template <typename T, bool UNGUARDED_IO, unsigned int SETS_PER_PASS, typename PreprocessFunctor>
__device__ __forceinline__ void ReadSets(
	typename VecType<T, 2>::Type *d_in, 
	typename VecType<T, 2>::Type pairs[SETS_PER_PASS],
	const unsigned int BASE2,
	int extra[1],
	PreprocessFunctor preprocess = PreprocessFunctor())				
{
	if (UNGUARDED_IO) {

		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler makes it 1% slower
		if (SETS_PER_PASS > 0) pairs[0] = d_in[threadIdx.x + BASE2 + (SRTS_THREADS * 0)];
		if (SETS_PER_PASS > 1) pairs[1] = d_in[threadIdx.x + BASE2 + (SRTS_THREADS * 1)];
		if (SETS_PER_PASS > 2) pairs[2] = d_in[threadIdx.x + BASE2 + (SRTS_THREADS * 2)];
		if (SETS_PER_PASS > 3) pairs[3] = d_in[threadIdx.x + BASE2 + (SRTS_THREADS * 3)];

	} else {

		T* in = (T*) d_in;
		
		// N.B. --  I wish we could do some pragma unrolling here, but the compiler won't let 
		// us with user-defined value types (e.g., Fribbitz): "Advisory: Loop was not unrolled, cannot deduce loop trip count"
		
		if (SETS_PER_PASS > 0) GuardedReadSet<T>(in, pairs[0], (threadIdx.x << 1) + (BASE2 << 1) + (SRTS_THREADS * 2 * 0), extra);
		if (SETS_PER_PASS > 1) GuardedReadSet<T>(in, pairs[1], (threadIdx.x << 1) + (BASE2 << 1) + (SRTS_THREADS * 2 * 1), extra);
		if (SETS_PER_PASS > 2) GuardedReadSet<T>(in, pairs[2], (threadIdx.x << 1) + (BASE2 << 1) + (SRTS_THREADS * 2 * 2), extra);
		if (SETS_PER_PASS > 3) GuardedReadSet<T>(in, pairs[3], (threadIdx.x << 1) + (BASE2 << 1) + (SRTS_THREADS * 2 * 3), extra);
		
	}
	
	#pragma unroll 
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		preprocess(pairs[SET].x);
		preprocess(pairs[SET].y);
	}
}


template <unsigned int SETS_PER_PASS>
__device__ __forceinline__ void PlacePartials(
	unsigned char * base_partial,
	uint2 digits[SETS_PER_PASS],
	uint2 flag_offsets[SETS_PER_PASS]) 
{
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		base_partial[flag_offsets[SET].x] = 1;
		base_partial[flag_offsets[SET].y] = 1 + (digits[SET].x == digits[SET].y);
	}
}


template <unsigned int SETS_PER_PASS>
__device__ __forceinline__ void ExtractRanks(
	unsigned char * base_partial,
	uint2 digits[SETS_PER_PASS],
	uint2 flag_offsets[SETS_PER_PASS],
	uint2 ranks[SETS_PER_PASS]) 
{
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		ranks[SET].x = base_partial[flag_offsets[SET].x];
		ranks[SET].y = base_partial[flag_offsets[SET].y] + (digits[SET].x == digits[SET].y);
	}
}


template <unsigned int RADIX_DIGITS, unsigned int SETS_PER_PASS>
__device__ __forceinline__ void UpdateRanks(
	uint2 digits[SETS_PER_PASS],
	uint2 ranks[SETS_PER_PASS],
	unsigned int digit_counts[SETS_PER_PASS][RADIX_DIGITS])
{
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		ranks[SET].x += digit_counts[SET][digits[SET].x];
		ranks[SET].y += digit_counts[SET][digits[SET].y]; 
	}
}



template <unsigned int SCAN_LANES_PER_PASS, unsigned int LOG_RAKING_THREADS_PER_LANE, unsigned int RAKING_THREADS_PER_LANE, unsigned int PARTIALS_PER_SEG>
__device__ __forceinline__ void PrefixScanOverLanes(
	unsigned int 	raking_segment[],
	unsigned int 	warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	unsigned int 	copy_section)
{
	// Upsweep rake
	unsigned int partial_reduction = SerialReduce<PARTIALS_PER_SEG>(raking_segment);

	// Warpscan reduction in digit warpscan_lane
	unsigned int warpscan_lane = threadIdx.x >> LOG_RAKING_THREADS_PER_LANE;
	unsigned int group_prefix = WarpScan<RAKING_THREADS_PER_LANE, true>(
		warpscan[warpscan_lane], 
		partial_reduction,
		copy_section);

	// Downsweep rake
	SerialScan<PARTIALS_PER_SEG>(raking_segment, group_prefix);
	
}


template <unsigned int SCAN_LANES_PER_PASS, unsigned int RAKING_THREADS_PER_LANE, unsigned int SETS_PER_PASS, unsigned int SCAN_LANES_PER_SET>
__device__ __forceinline__ void RecoverDigitCounts(
	unsigned int warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	unsigned int counts[SETS_PER_PASS],
	unsigned int copy_section)
{
	unsigned int my_lane = threadIdx.x >> 2;
	unsigned int my_quad_byte = threadIdx.x & 3;
	
	#pragma unroll
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		unsigned char *warpscan_count = (unsigned char *) &warpscan[my_lane + (SCAN_LANES_PER_SET * SET)][1 + copy_section][RAKING_THREADS_PER_LANE - 1];
		counts[SET] = warpscan_count[my_quad_byte];
	}
}


__device__ __forceinline__ void CorrectUnguardedSetOverflow(
	uint2 			set_digits,
	unsigned int 	&set_count)				
{
	if (WarpVoteAll(set_count <= 1)) {
		// All first-pass, first set keys have same digit. 
		set_count = (threadIdx.x == set_digits.x) ? 256 : 0;
	}
}

template <unsigned int SETS_PER_PASS>
__device__ __forceinline__ void CorrectUnguardedPassOverflow(
	uint2 			pass_digits[SETS_PER_PASS],
	unsigned int 	pass_counts[SETS_PER_PASS])				
{
	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

	if (SETS_PER_PASS > 0) CorrectUnguardedSetOverflow(pass_digits[0], pass_counts[0]);
	if (SETS_PER_PASS > 1) CorrectUnguardedSetOverflow(pass_digits[1], pass_counts[1]);
	if (SETS_PER_PASS > 2) CorrectUnguardedSetOverflow(pass_digits[2], pass_counts[2]);
	if (SETS_PER_PASS > 3) CorrectUnguardedSetOverflow(pass_digits[3], pass_counts[3]);
}


template <unsigned int PASSES_PER_CYCLE, unsigned int SETS_PER_PASS>
__device__ __forceinline__ void CorrectUnguardedCycleOverflow(
	uint2 			cycle_digits[PASSES_PER_CYCLE][SETS_PER_PASS],
	unsigned int 	cycle_counts[PASSES_PER_CYCLE][SETS_PER_PASS])
{
	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

	if (PASSES_PER_CYCLE > 0) CorrectUnguardedPassOverflow<SETS_PER_PASS>(cycle_digits[0], cycle_counts[0]);
	if (PASSES_PER_CYCLE > 1) CorrectUnguardedPassOverflow<SETS_PER_PASS>(cycle_digits[1], cycle_counts[1]);
}


template <unsigned int RADIX_DIGITS>
__device__ __forceinline__ void CorrectLastLaneOverflow(unsigned int &count, int extra[1]) 
{
	if (WarpVoteAll(count == 0) && (threadIdx.x == RADIX_DIGITS - 1)) {
		// We're 'f' and we overflowed b/c of invalid 'f' placemarkers; the number of valid items in this set is the count of valid f's 
		count = extra[0] & 255;
	}
}
		

template <unsigned int RADIX_DIGITS, unsigned int PASSES_PER_CYCLE, unsigned int SETS_PER_PASS, unsigned int SETS_PER_CYCLE, bool UNGUARDED_IO>
__device__ __forceinline__ void CorrectForOverflows(
	uint2 digits[PASSES_PER_CYCLE][SETS_PER_PASS],
	unsigned int counts[PASSES_PER_CYCLE][SETS_PER_PASS], 
	int extra[1])				
{
	if (!UNGUARDED_IO) {
		
		unsigned int *linear_counts = (unsigned int *) counts;
		
		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected call OPs"

		if (SETS_PER_CYCLE > 0) CorrectLastLaneOverflow<RADIX_DIGITS>(linear_counts[0], extra);
		if (SETS_PER_CYCLE > 1) if (extra[0] < 1 * 256) CorrectLastLaneOverflow<RADIX_DIGITS>(linear_counts[1], extra);
		if (SETS_PER_CYCLE > 2) if (extra[0] < 2 * 256) CorrectLastLaneOverflow<RADIX_DIGITS>(linear_counts[2], extra);
		if (SETS_PER_CYCLE > 3) if (extra[0] < 3 * 256) CorrectLastLaneOverflow<RADIX_DIGITS>(linear_counts[3], extra);
	}

	CorrectUnguardedCycleOverflow<PASSES_PER_CYCLE, SETS_PER_PASS>(digits, counts);
}


template <
	typename K,
	unsigned int BIT, 
	unsigned int RADIX_DIGITS,
	unsigned int SCAN_LANES_PER_SET,
	unsigned int SETS_PER_PASS,
	unsigned int RAKING_THREADS_PER_PASS,
	unsigned int SCAN_LANES_PER_PASS,
	unsigned int LOG_RAKING_THREADS_PER_LANE,
	unsigned int RAKING_THREADS_PER_LANE,
	unsigned int PARTIALS_PER_SEG,
	unsigned int PADDED_PARTIALS_PER_LANE,
	unsigned int PASSES_PER_CYCLE>
__device__ __forceinline__ void ScanPass(
	unsigned int 					*base_partial,
	unsigned int					*raking_partial,
	unsigned int 					warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	typename VecType<K, 2>::Type 	keypairs[SETS_PER_PASS],
	uint2 							digits[SETS_PER_PASS],
	uint2 							flag_offsets[SETS_PER_PASS],
	uint2							ranks[SETS_PER_PASS],
	unsigned int 					copy_section)
{
	// Reset smem
	#pragma unroll
	for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_PASS; SCAN_LANE++) {
		base_partial[SCAN_LANE * PADDED_PARTIALS_PER_LANE] = 0;
	}
	
	// Decode digits for first pass
	DecodeDigits<K, RADIX_DIGITS, BIT, SETS_PER_PASS, SCAN_LANES_PER_SET, PADDED_PARTIALS_PER_LANE>(
		keypairs, digits, flag_offsets);
	
	// Encode counts into smem for first pass
	PlacePartials<SETS_PER_PASS>(
		(unsigned char *) base_partial,
		digits,
		flag_offsets); 
	
	__syncthreads();
	
	// Intra-group prefix scans for first pass
	if (threadIdx.x < RAKING_THREADS_PER_PASS) {
	
		PrefixScanOverLanes<SCAN_LANES_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG>(		// first pass is offset right by one
			raking_partial,
			warpscan, 
			copy_section);
	}
	
	__syncthreads();

	// Extract ranks
	ExtractRanks<SETS_PER_PASS>(
		(unsigned char *) base_partial, 
		digits, 
		flag_offsets, 
		ranks); 	
}	
	

//------------------------------------------------------------------------------
// SM1.3 Local Exchange Routines
//
// Routines for exchanging keys (and values) in shared memory (i.e., local 
// scattering) in order to to facilitate coalesced global scattering
//------------------------------------------------------------------------------

template <typename T, bool UNGUARDED_IO, unsigned int PASSES_PER_CYCLE, unsigned int SETS_PER_PASS, typename PostprocessFunctor>
__device__ __forceinline__ void ScatterSets(
	T *d_out, 
	typename VecType<T, 2>::Type pairs[SETS_PER_PASS],
	uint2 offsets[SETS_PER_PASS],
	const unsigned int BASE4,
	int extra[1],
	PostprocessFunctor postprocess = PostprocessFunctor())				
{
	#pragma unroll 
	for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
		postprocess(pairs[SET].x);
		postprocess(pairs[SET].y);
	}

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler makes it 1% slower 
		
	if (SETS_PER_PASS > 0) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 0) < extra[0])) 
			d_out[offsets[0].x] = pairs[0].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 1) < extra[0])) 
			d_out[offsets[0].y] = pairs[0].y;
	}

	if (SETS_PER_PASS > 1) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 2) < extra[0])) 
			d_out[offsets[1].x] = pairs[1].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 3) < extra[0])) 
			d_out[offsets[1].y] = pairs[1].y;
	}

	if (SETS_PER_PASS > 2) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 4) < extra[0])) 
			d_out[offsets[2].x] = pairs[2].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 5) < extra[0])) 
			d_out[offsets[2].y] = pairs[2].y;
	}

	if (SETS_PER_PASS > 3) { 
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 6) < extra[0])) 
			d_out[offsets[3].x] = pairs[3].x;
		if (UNGUARDED_IO || (threadIdx.x + BASE4 + (SRTS_THREADS * 7) < extra[0])) 
			d_out[offsets[3].y] = pairs[3].y;
	}
}

template <typename T, unsigned int PASSES_PER_CYCLE, unsigned int SETS_PER_PASS>
__device__ __forceinline__ void PushPairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[PASSES_PER_CYCLE][SETS_PER_PASS],
	uint2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS])				
{
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
	
		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			swap[ranks[PASS][SET].x] = pairs[PASS][SET].x;
			swap[ranks[PASS][SET].y] = pairs[PASS][SET].y;
		}
	}
}
	
template <typename T, unsigned int PASSES_PER_CYCLE, unsigned int SETS_PER_PASS>
__device__ __forceinline__ void ExchangePairs(
	T *swap, 
	typename VecType<T, 2>::Type pairs[PASSES_PER_CYCLE][SETS_PER_PASS],
	uint2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS])				
{
	// Push in Pairs
	PushPairs<T, PASSES_PER_CYCLE, SETS_PER_PASS>(swap, pairs, ranks);
	
	__syncthreads();
	
	// Extract pairs
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		
		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			const int BLOCK = ((PASS * SETS_PER_PASS) + SET) * 2;
			pairs[PASS][SET].x = swap[threadIdx.x + (SRTS_THREADS * (BLOCK + 0))];
			pairs[PASS][SET].y = swap[threadIdx.x + (SRTS_THREADS * (BLOCK + 1))];
		}
	}
}


template <
	typename K,
	typename V,	
	bool KEYS_ONLY, 
	unsigned int RADIX_DIGITS, 
	unsigned int BIT, 
	unsigned int PASSES_PER_CYCLE,
	unsigned int SETS_PER_PASS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterSm13(
	typename VecType<K, 2>::Type keypairs[PASSES_PER_CYCLE][SETS_PER_PASS], 
	uint2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	unsigned int *exchange,
	typename VecType<V, 2>::Type *d_in_data, 
	K *d_out_keys, 
	V *d_out_data, 
	unsigned int carry[RADIX_DIGITS], 
	int extra[1])				
{
	uint2 offsets[PASSES_PER_CYCLE][SETS_PER_PASS];
	
	// Swap keys according to ranks
	ExchangePairs<K, PASSES_PER_CYCLE, SETS_PER_PASS>((K*) exchange, keypairs, ranks);				
	
	// Calculate scatter offsets (re-decode digits from keys: it's less work than making a second exchange of digits) 
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		
		#pragma unroll 
		for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
			const int BLOCK = ((PASS * SETS_PER_PASS) + SET) * 2;
			offsets[PASS][SET].x = threadIdx.x + (SRTS_THREADS * (BLOCK + 0)) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypairs[PASS][SET].x)];
			offsets[PASS][SET].y = threadIdx.x + (SRTS_THREADS * (BLOCK + 1)) + carry[DecodeDigit<K, RADIX_DIGITS, BIT>(keypairs[PASS][SET].y)];
		}
	}
	
	// Scatter keys
	#pragma unroll 
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		const int BLOCK = PASS * SETS_PER_PASS * 2;
		ScatterSets<K, UNGUARDED_IO, PASSES_PER_CYCLE, SETS_PER_PASS, PostprocessFunctor>(d_out_keys, keypairs[PASS], offsets[PASS], SRTS_THREADS * BLOCK, extra);
	}

	if (!KEYS_ONLY) {
	
		__syncthreads();

		// Read input data
		typename VecType<V, 2>::Type datapairs[PASSES_PER_CYCLE][SETS_PER_PASS];

		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		if (PASSES_PER_CYCLE > 0) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_data, datapairs[0], SRTS_THREADS * SETS_PER_PASS * 0, extra);
		if (PASSES_PER_CYCLE > 1) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_data, datapairs[1], SRTS_THREADS * SETS_PER_PASS * 1, extra);
		
		// Swap data according to ranks
		ExchangePairs<V, PASSES_PER_CYCLE, SETS_PER_PASS>((V*) exchange, datapairs, ranks);
		
		// Scatter data
		#pragma unroll 
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
			const int BLOCK = PASS * SETS_PER_PASS * 2;
			ScatterSets<V, UNGUARDED_IO, PASSES_PER_CYCLE, SETS_PER_PASS, NopFunctor<V> >(d_out_data, datapairs[PASS], offsets[PASS], SRTS_THREADS * BLOCK, extra);
		}
	}
}


//------------------------------------------------------------------------------
// SM1.0 Local Exchange Routines
//
// Routines for exchanging keys (and values) in shared memory (i.e., local 
// scattering) in order to to facilitate coalesced global scattering
//------------------------------------------------------------------------------

template <
	typename T, 
	unsigned int RADIX_DIGITS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor> 
__device__ __forceinline__ void ScatterPass(
	T *swapmem,
	T *d_out, 
	unsigned int digit_scan[2][RADIX_DIGITS], 
	unsigned int carry[RADIX_DIGITS], 
	int extra[1],
	unsigned int base_digit)				
{
	const unsigned int LOG_HALF_WARP_THREADS = LOG_WARP_THREADS - 1;
	const unsigned int HALF_WARP_THREADS = 1 << LOG_HALF_WARP_THREADS;
	
	int half_warp_idx = threadIdx.x & (HALF_WARP_THREADS - 1);
	int half_warp_digit = threadIdx.x >> LOG_HALF_WARP_THREADS;
	
	int my_digit = base_digit + half_warp_digit;
	if (my_digit < RADIX_DIGITS) {
	
		int my_exclusive_scan = digit_scan[1][my_digit - 1];
		int my_inclusive_scan = digit_scan[1][my_digit];
		int my_digit_count = my_inclusive_scan - my_exclusive_scan;

		int my_carry = carry[my_digit] + my_exclusive_scan;
		int my_aligned_offset = half_warp_idx - (my_carry & (HALF_WARP_THREADS - 1));
		
		while (my_aligned_offset < my_digit_count) {
			if ((my_aligned_offset >= 0) && (UNGUARDED_IO || (my_exclusive_scan + my_aligned_offset < extra[0]))) { 
				d_out[my_carry + my_aligned_offset] = swapmem[my_exclusive_scan + my_aligned_offset];
			}
			my_aligned_offset += 16;
		}
	}
}

template <
	typename T,
	unsigned int RADIX_DIGITS, 
	unsigned int PASSES_PER_CYCLE,
	unsigned int SETS_PER_PASS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterPairs(
	typename VecType<T, 2>::Type pairs[PASSES_PER_CYCLE][SETS_PER_PASS], 
	uint2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	T *exchange,
	T *d_out, 
	unsigned int carry[RADIX_DIGITS], 
	unsigned int digit_scan[2][RADIX_DIGITS], 
	int extra[1])				
{
	const unsigned int SCATTER_PASS_DIGITS = SRTS_WARPS * 2;
	const unsigned int SCATTER_PASSES = RADIX_DIGITS / SCATTER_PASS_DIGITS;

	// Push in pairs
	PushPairs<T, PASSES_PER_CYCLE, SETS_PER_PASS>(exchange, pairs, ranks);

	__syncthreads();

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, not an innermost loop"

	if (SCATTER_PASSES > 0) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 0);
	if (SCATTER_PASSES > 1) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 1);
	if (SCATTER_PASSES > 2) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 2);
	if (SCATTER_PASSES > 3) ScatterPass<T, RADIX_DIGITS, UNGUARDED_IO, PostprocessFunctor>(exchange, d_out, digit_scan, carry, extra, SCATTER_PASS_DIGITS * 3);
}


template <
	typename K,
	typename V,	
	bool KEYS_ONLY, 
	unsigned int RADIX_DIGITS, 
	unsigned int PASSES_PER_CYCLE,
	unsigned int SETS_PER_PASS,
	bool UNGUARDED_IO,
	typename PostprocessFunctor>
__device__ __forceinline__ void SwapAndScatterSm10(
	typename VecType<K, 2>::Type keypairs[PASSES_PER_CYCLE][SETS_PER_PASS], 
	uint2 ranks[PASSES_PER_CYCLE][SETS_PER_PASS],
	unsigned int *exchange,
	typename VecType<V, 2>::Type *d_in_data, 
	K *d_out_keys, 
	V *d_out_data, 
	unsigned int carry[RADIX_DIGITS], 
	unsigned int digit_scan[2][RADIX_DIGITS], 
	int extra[1])				
{
	// Swap and scatter keys
	SwapAndScatterPairs<K, RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, ranks, (K*) exchange, d_out_keys, carry, digit_scan, extra);				
	
	if (!KEYS_ONLY) {

		__syncthreads();
		
		// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
		// telling me "Advisory: Loop was not unrolled, unexpected control flow"

		// Read input data
		typename VecType<V, 2>::Type datapairs[PASSES_PER_CYCLE][SETS_PER_PASS];
		if (PASSES_PER_CYCLE > 0) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_data, datapairs[0], SRTS_THREADS * SETS_PER_PASS * 0, extra);
		if (PASSES_PER_CYCLE > 1) ReadSets<V, UNGUARDED_IO, SETS_PER_PASS, NopFunctor<V> >(d_in_data, datapairs[1], SRTS_THREADS * SETS_PER_PASS * 1, extra);

		// Swap and scatter data
		SwapAndScatterPairs<V, RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
			datapairs, ranks, (V*) exchange, d_out_data, carry, digit_scan, extra);				
	}
}


//------------------------------------------------------------------------------
// Cycle of SRTS_CYCLE_ELEMENTS keys (and values)
//------------------------------------------------------------------------------

template <
	typename K,
	typename V,	
	bool KEYS_ONLY, 
	unsigned int BIT, 
	bool UNGUARDED_IO,
	unsigned int RADIX_DIGITS,
	unsigned int LOG_SCAN_LANES_PER_SET,
	unsigned int SCAN_LANES_PER_SET,
	unsigned int SETS_PER_PASS,
	unsigned int PASSES_PER_CYCLE,
	unsigned int LOG_SCAN_LANES_PER_PASS,
	unsigned int SCAN_LANES_PER_PASS,
	unsigned int LOG_PARTIALS_PER_LANE,
	unsigned int LOG_PARTIALS_PER_PASS,
	unsigned int LOG_RAKING_THREADS_PER_PASS,
	unsigned int RAKING_THREADS_PER_PASS,
	unsigned int LOG_RAKING_THREADS_PER_LANE,
	unsigned int RAKING_THREADS_PER_LANE,
	unsigned int LOG_PARTIALS_PER_SEG,
	unsigned int PARTIALS_PER_SEG,
	unsigned int LOG_PARTIALS_PER_ROW,
	unsigned int PARTIALS_PER_ROW,
	unsigned int LOG_SEGS_PER_ROW,	
	unsigned int SEGS_PER_ROW,
	unsigned int LOG_ROWS_PER_SET,
	unsigned int LOG_ROWS_PER_LANE,
	unsigned int ROWS_PER_LANE,
	unsigned int LOG_ROWS_PER_PASS,
	unsigned int ROWS_PER_PASS,
	unsigned int MAX_EXCHANGE_BYTES,
	typename PreprocessFunctor,
	typename PostprocessFunctor>

__device__ __forceinline__ void SrtsScanDigitCycle(
	typename VecType<K, 2>::Type 	*d_in_keys, 
	typename VecType<V, 2>::Type 	*d_in_data, 
	K								*d_out_keys, 
	V								*d_out_data, 
	unsigned int 					*exchange,								
	unsigned int 					warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE],
	unsigned int 					carry[RADIX_DIGITS],
	unsigned int 					digit_scan[2][RADIX_DIGITS],						 
	unsigned int 					digit_counts[PASSES_PER_CYCLE][SETS_PER_PASS][RADIX_DIGITS],
	int 							extra[1],
	unsigned int 					*base_partial,
	unsigned int 					*raking_partial)		
{
	
	const unsigned int PADDED_PARTIALS_PER_LANE 		= ROWS_PER_LANE * (PARTIALS_PER_ROW + 1);	// N.B.: we have "declared but never referenced" warnings for these, but they're actually used for template instantiation 
	const unsigned int SETS_PER_CYCLE 					= PASSES_PER_CYCLE * SETS_PER_PASS;
	
	typename VecType<K, 2>::Type 	keypairs[PASSES_PER_CYCLE][SETS_PER_PASS];
	uint2 							digits[PASSES_PER_CYCLE][SETS_PER_PASS];
	uint2 							flag_offsets[PASSES_PER_CYCLE][SETS_PER_PASS];		// a byte offset
	uint2 							ranks[PASSES_PER_CYCLE][SETS_PER_PASS];

	
	//-------------------------------------------------------------------------
	// Read keys
	//-------------------------------------------------------------------------

	// N.B. -- I wish we could do some pragma unrolling here too, but the compiler won't comply, 
	// telling me "Advisory: Loop was not unrolled, unexpected control flow construct"
	
	// Read Keys
	if (PASSES_PER_CYCLE > 0) ReadSets<K, UNGUARDED_IO, SETS_PER_PASS, PreprocessFunctor>(d_in_keys, keypairs[0], SRTS_THREADS * SETS_PER_PASS * 0, extra);		 
	if (PASSES_PER_CYCLE > 1) ReadSets<K, UNGUARDED_IO, SETS_PER_PASS, PreprocessFunctor>(d_in_keys, keypairs[1], SRTS_THREADS * SETS_PER_PASS * 1, extra); 	
	
	//-------------------------------------------------------------------------
	// Lane-scanning Passes
	//-------------------------------------------------------------------------

	#pragma unroll
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
	
		// First Pass
		ScanPass<K, BIT, RADIX_DIGITS, SCAN_LANES_PER_SET, SETS_PER_PASS, RAKING_THREADS_PER_PASS, SCAN_LANES_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, PARTIALS_PER_SEG, PADDED_PARTIALS_PER_LANE, PASSES_PER_CYCLE>(
			base_partial,
			raking_partial,
			warpscan,
			keypairs[PASS],
			digits[PASS],
			flag_offsets[PASS],
			ranks[PASS],
			PASSES_PER_CYCLE - PASS - 1);		// lower passes get copied right
	}
	
	//-------------------------------------------------------------------------
	// Digit-scanning 
	//-------------------------------------------------------------------------

	// Recover second-half digit-counts, scan across all digit-counts
	if (threadIdx.x < RADIX_DIGITS) {

		unsigned int counts[PASSES_PER_CYCLE][SETS_PER_PASS];

		// Recover digit-counts

		#pragma unroll
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
			RecoverDigitCounts<SCAN_LANES_PER_PASS, RAKING_THREADS_PER_LANE, SETS_PER_PASS, SCAN_LANES_PER_SET>(		// first pass, offset by 1			
				warpscan, 
				counts[PASS],
				PASSES_PER_CYCLE - PASS - 1);		// lower passes get copied right
		}
		
		// Check for overflows
		CorrectForOverflows<RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, SETS_PER_CYCLE, UNGUARDED_IO>(
				digits, counts, extra);

		// Scan across my digit counts for each set 
		unsigned int exclusive_total = 0;
		unsigned int inclusive_total = 0;
		
		#pragma unroll
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		
			#pragma unroll
			for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
				inclusive_total += counts[PASS][SET];
				counts[PASS][SET] = exclusive_total;
				exclusive_total = inclusive_total;
			}
		}

		// second half of carry update
		unsigned int my_carry = carry[threadIdx.x] + digit_scan[1][threadIdx.x];

		// Perform overflow-free SIMD Kogge-Stone across digits
		unsigned int digit_prefix = WarpScan<RADIX_DIGITS, false>(
				digit_scan, 
				inclusive_total,
				0);

		// first-half of carry update 
		carry[threadIdx.x] = my_carry - digit_prefix;
		
		#pragma unroll
		for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {

			#pragma unroll
			for (int SET = 0; SET < (int) SETS_PER_PASS; SET++) {
				digit_counts[PASS][SET][threadIdx.x] = counts[PASS][SET] + digit_prefix;
			}
		}
	}
	
	__syncthreads();

	//-------------------------------------------------------------------------
	// Update Ranks
	//-------------------------------------------------------------------------

	#pragma unroll
	for (int PASS = 0; PASS < (int) PASSES_PER_CYCLE; PASS++) {
		UpdateRanks<RADIX_DIGITS, SETS_PER_PASS>(digits[PASS], ranks[PASS], digit_counts[PASS]);
	}
	
	
	//-------------------------------------------------------------------------
	// Scatter 
	//-------------------------------------------------------------------------

#if __CUDA_ARCH__ < 130		

	SwapAndScatterSm10<K, V, KEYS_ONLY, RADIX_DIGITS, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_data, 
		d_out_keys, 
		d_out_data, 
		carry, 
		digit_scan,
		extra);
	
#else 

	SwapAndScatterSm13<K, V, KEYS_ONLY, RADIX_DIGITS, BIT, PASSES_PER_CYCLE, SETS_PER_PASS, UNGUARDED_IO, PostprocessFunctor>(
		keypairs, 
		ranks,
		exchange,
		d_in_data, 
		d_out_keys, 
		d_out_data, 
		carry, 
		extra);
	
#endif

	__syncthreads();

}



//------------------------------------------------------------------------------
// Scan/Scatter Kernel Entry Point
//------------------------------------------------------------------------------

template <typename K, typename V, bool KEYS_ONLY, unsigned int RADIX_BITS, unsigned int BIT, typename PreprocessFunctor, typename PostprocessFunctor>
__launch_bounds__ (SRTS_THREADS, SRTS_BULK_CTA_OCCUPANCY(__CUDA_ARCH__))
__global__ 
void SrtsScanDigitBulk(
	unsigned int* d_spine,
	K* d_in_keys,
	K* d_out_keys,
	V* d_in_data,
	V* d_out_data,
	CtaDecomposition work_decomposition)
{

	const unsigned int RADIX_DIGITS 			= 1 << RADIX_BITS;
	
	const unsigned int LOG_SCAN_LANES_PER_SET	= (RADIX_BITS > 2) ? RADIX_BITS - 2 : 0;					// Always at one lane per set
	const unsigned int SCAN_LANES_PER_SET		= 1 << LOG_SCAN_LANES_PER_SET;								// N.B.: we have "declared but never referenced" warnings for these, but they're actually used for template instantiation
	
	const unsigned int LOG_SETS_PER_PASS		= SRTS_LOG_SETS_PER_PASS(__CUDA_ARCH__);			
	const unsigned int SETS_PER_PASS			= 1 << LOG_SETS_PER_PASS;
	
	const unsigned int LOG_PASSES_PER_CYCLE		= SRTS_LOG_PASSES_PER_CYCLE(__CUDA_ARCH__, K, V);			
	const unsigned int PASSES_PER_CYCLE			= 1 << LOG_PASSES_PER_CYCLE;

	const unsigned int LOG_SCAN_LANES_PER_PASS	= LOG_SETS_PER_PASS + LOG_SCAN_LANES_PER_SET;
	const unsigned int SCAN_LANES_PER_PASS		= 1 << LOG_SCAN_LANES_PER_PASS;
	
	const unsigned int LOG_PARTIALS_PER_LANE 	= SRTS_LOG_THREADS;
	
	const unsigned int LOG_PARTIALS_PER_PASS	= LOG_SCAN_LANES_PER_PASS + LOG_PARTIALS_PER_LANE;

	const unsigned int LOG_RAKING_THREADS_PER_PASS 		= SRTS_LOG_RAKING_THREADS_PER_PASS(__CUDA_ARCH__);
	const unsigned int RAKING_THREADS_PER_PASS			= 1 << LOG_RAKING_THREADS_PER_PASS;

	const unsigned int LOG_RAKING_THREADS_PER_LANE 		= LOG_RAKING_THREADS_PER_PASS - LOG_SCAN_LANES_PER_PASS;
	const unsigned int RAKING_THREADS_PER_LANE 			= 1 << LOG_RAKING_THREADS_PER_LANE;

	const unsigned int LOG_PARTIALS_PER_SEG 	= LOG_PARTIALS_PER_LANE - LOG_RAKING_THREADS_PER_LANE;
	const unsigned int PARTIALS_PER_SEG 		= 1 << LOG_PARTIALS_PER_SEG;

	const unsigned int LOG_PARTIALS_PER_ROW		= (LOG_PARTIALS_PER_SEG < LOG_MEM_BANKS(__CUDA_ARCH__)) ? LOG_MEM_BANKS(__CUDA_ARCH__) : LOG_PARTIALS_PER_SEG;		// floor of MEM_BANKS partials per row
	const unsigned int PARTIALS_PER_ROW			= 1 << LOG_PARTIALS_PER_ROW;
	const unsigned int PADDED_PARTIALS_PER_ROW 	= PARTIALS_PER_ROW + 1;

	const unsigned int LOG_SEGS_PER_ROW 		= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;	
	const unsigned int SEGS_PER_ROW				= 1 << LOG_SEGS_PER_ROW;

	const unsigned int LOG_ROWS_PER_SET 		= LOG_PARTIALS_PER_PASS - LOG_PARTIALS_PER_ROW;

	const unsigned int LOG_ROWS_PER_LANE 		= LOG_PARTIALS_PER_LANE - LOG_PARTIALS_PER_ROW;
	const unsigned int ROWS_PER_LANE 			= 1 << LOG_ROWS_PER_LANE;

	const unsigned int LOG_ROWS_PER_PASS 		= LOG_SCAN_LANES_PER_PASS + LOG_ROWS_PER_LANE;
	const unsigned int ROWS_PER_PASS 			= 1 << LOG_ROWS_PER_PASS;
	
	const unsigned int SCAN_LANE_BYTES			= ROWS_PER_PASS * PADDED_PARTIALS_PER_ROW * sizeof(unsigned int);
	const unsigned int MAX_EXCHANGE_BYTES		= (sizeof(K) > sizeof(V)) ? 
													SRTS_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V) * sizeof(K) : 
													SRTS_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V) * sizeof(V);
	const unsigned int EXCHANGE_PADDING_QUADS	= (MAX_EXCHANGE_BYTES > SCAN_LANE_BYTES) ? (MAX_EXCHANGE_BYTES - SCAN_LANE_BYTES + sizeof(unsigned int) - 1) / sizeof(unsigned int) : 0;

	
	__shared__ unsigned int 	scan_lanes[(ROWS_PER_PASS * PADDED_PARTIALS_PER_ROW) + EXCHANGE_PADDING_QUADS];
	__shared__ unsigned int 	warpscan[SCAN_LANES_PER_PASS][3][RAKING_THREADS_PER_LANE];		// One warpscan per fours-group
	__shared__ unsigned int 	carry[RADIX_DIGITS];
	__shared__ unsigned int 	digit_scan[2][RADIX_DIGITS];						 
	__shared__ unsigned int 	digit_counts[PASSES_PER_CYCLE][SETS_PER_PASS][RADIX_DIGITS];

	REG_MISER_QUALIFIER int	extra[1];
	REG_MISER_QUALIFIER int oob[1];

	
	extra[0] = (blockIdx.x == gridDim.x - 1) ? work_decomposition.extra_elements_last_block : 0;

	// calculate our threadblock's range
	unsigned int block_elements, block_offset;
	if (blockIdx.x < work_decomposition.num_big_blocks) {
		block_offset = work_decomposition.big_block_elements * blockIdx.x;
		block_elements = work_decomposition.big_block_elements;
	} else {
		block_offset = (work_decomposition.normal_block_elements * blockIdx.x) + (work_decomposition.num_big_blocks * SRTS_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V));
		block_elements = work_decomposition.normal_block_elements;
	}
	oob[0] = block_offset + block_elements;	// out-of-bounds

	
	// location for placing 2-element partial reductions in the first lane of a pass	
	unsigned int row = threadIdx.x >> LOG_PARTIALS_PER_ROW; 
	unsigned int col = threadIdx.x & (PARTIALS_PER_ROW - 1); 
	unsigned int *base_partial = scan_lanes + (row * PADDED_PARTIALS_PER_ROW) + col; 								
	
	// location for raking across all sets within a pass
	unsigned int *raking_partial;										

	if (threadIdx.x < RAKING_THREADS_PER_PASS) {

		// initalize lane warpscans
		if (threadIdx.x < RAKING_THREADS_PER_LANE) {
			
			#pragma unroll
			for (int SCAN_LANE = 0; SCAN_LANE < (int) SCAN_LANES_PER_PASS; SCAN_LANE++) {
				warpscan[SCAN_LANE][0][threadIdx.x] = 0;
			}
		}

		// initialize digit warpscans
		if (threadIdx.x < RADIX_DIGITS) {

			// read carry in parallel 
			carry[threadIdx.x] = d_spine[(gridDim.x * threadIdx.x) + blockIdx.x];

			// initialize digit_scan
			digit_scan[0][threadIdx.x] = 0;
			digit_scan[1][threadIdx.x] = 0;
		}

		// initialize raking segment
		row = threadIdx.x >> LOG_SEGS_PER_ROW;
		col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		raking_partial = scan_lanes + (row * PADDED_PARTIALS_PER_ROW) + col; 
	}

	// Scan in tiles of cycle_elements
	while (block_offset < oob[0]) {

		SrtsScanDigitCycle<K, V, KEYS_ONLY, BIT, true, RADIX_DIGITS, LOG_SCAN_LANES_PER_SET, SCAN_LANES_PER_SET, SETS_PER_PASS, PASSES_PER_CYCLE, LOG_SCAN_LANES_PER_PASS, SCAN_LANES_PER_PASS, LOG_PARTIALS_PER_LANE, LOG_PARTIALS_PER_PASS, LOG_RAKING_THREADS_PER_PASS, RAKING_THREADS_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, LOG_PARTIALS_PER_SEG, PARTIALS_PER_SEG, LOG_PARTIALS_PER_ROW, PARTIALS_PER_ROW, LOG_SEGS_PER_ROW, SEGS_PER_ROW, LOG_ROWS_PER_SET, LOG_ROWS_PER_LANE, ROWS_PER_LANE, LOG_ROWS_PER_PASS, ROWS_PER_PASS, MAX_EXCHANGE_BYTES, PreprocessFunctor, PostprocessFunctor>(	
			(typename VecType<K, 2>::Type *) &d_in_keys[block_offset], 
			(typename VecType<V, 2>::Type *) &d_in_data[block_offset], 
			d_out_keys, 
			d_out_data, 
			scan_lanes,
			warpscan,
			carry,
			digit_scan,						 
			digit_counts,
			extra,
			base_partial,
			raking_partial);		

		block_offset += SRTS_CYCLE_ELEMENTS(__CUDA_ARCH__, K, V);
	}

	if (extra[0]) {
		
		SrtsScanDigitCycle<K, V, KEYS_ONLY, BIT, false, RADIX_DIGITS, LOG_SCAN_LANES_PER_SET, SCAN_LANES_PER_SET, SETS_PER_PASS, PASSES_PER_CYCLE, LOG_SCAN_LANES_PER_PASS, SCAN_LANES_PER_PASS, LOG_PARTIALS_PER_LANE, LOG_PARTIALS_PER_PASS, LOG_RAKING_THREADS_PER_PASS, RAKING_THREADS_PER_PASS, LOG_RAKING_THREADS_PER_LANE, RAKING_THREADS_PER_LANE, LOG_PARTIALS_PER_SEG, PARTIALS_PER_SEG, LOG_PARTIALS_PER_ROW, PARTIALS_PER_ROW, LOG_SEGS_PER_ROW, SEGS_PER_ROW, LOG_ROWS_PER_SET, LOG_ROWS_PER_LANE, ROWS_PER_LANE, LOG_ROWS_PER_PASS, ROWS_PER_PASS, MAX_EXCHANGE_BYTES, PreprocessFunctor, PostprocessFunctor>(	
			(typename VecType<K, 2>::Type *) &d_in_keys[block_offset], 
			(typename VecType<V, 2>::Type *) &d_in_data[block_offset], 
			d_out_keys, 
			d_out_data, 
			scan_lanes,
			warpscan,
			carry,
			digit_scan,						 
			digit_counts,
			extra,
			base_partial,
			raking_partial);		
	}
}



#endif



