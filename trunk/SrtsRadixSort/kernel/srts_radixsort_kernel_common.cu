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
// Common SRTS Radix Sorting Properties and Routines 
//------------------------------------------------------------------------------


#ifndef _SRTS_RADIX_SORT_COMMON_KERNEL_H_
#define _SRTS_RADIX_SORT_COMMON_KERNEL_H_

#include <cuda.h>


//------------------------------------------------------------------------------
// Device properties 
//------------------------------------------------------------------------------


#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 0
#endif

#define FERMI(version)								(version >= 200)
#define LOG_WARP_THREADS							5									// 32 threads in a warp
#define WARP_THREADS								(1 << LOG_WARP_THREADS)
#define LOG_MEM_BANKS(version) 						((version >= 200) ? 5 : 4)			// 32 banks on fermi, 16 on tesla
#define MEM_BANKS(version)							(1 << LOG_MEM_BANKS(version))

#if __CUDA_ARCH__ >= 200
	#define FastMul(a, b) (a * b)
#else
	#define FastMul(a, b) (__umul24(a, b))
#endif	


#if __CUDA_ARCH__ >= 120
	#define WarpVoteAll(predicate) (__all(predicate))
#else 
	#define WarpVoteAll(predicate) (EmulatedWarpVoteAll(predicate))
#endif


#ifdef __LP64__
	#define SRTS_LP64 true
#else
	#define SRTS_LP64 false
#endif


//------------------------------------------------------------------------------
// Handy routines 
//------------------------------------------------------------------------------

#define MAX(a, b) ((a > b) ? a : b)


/**
 * Support structures for MagnitudeShift() below.  Allows you to shift 
 * left for positive magnitude values, right for negative.   
 * 
 * N.B. This code is a little strange; we are using this meta-programming 
 * pattern of partial template specialization for structures in order to 
 * decide whether to shift left or right.  Normally we would just use a 
 * conditional to decide if something was negative or not and then shift 
 * accordingly, knowing that the compiler will elide the untaken branch, 
 * i.e., the out-of-bounds shift during dead code elimination. However, 
 * the pass for bounds-checking shifts seems to happen before the DCE 
 * phase, which results in a an unsightly number of compiler warnings, so 
 * we force the issue earlier using structural template specialization.
 */

template <typename K, int magnitude, bool shift_left> struct MagnitudeShiftOp;

template <typename K, int magnitude> 
struct MagnitudeShiftOp<K, magnitude, true> {
	__device__ __forceinline__ static K Shift(K key) {
		return key << magnitude;
	}
};

template <typename K, int magnitude> 
struct MagnitudeShiftOp<K, magnitude, false> {
	__device__ __forceinline__ static K Shift(K key) {
		return key >> magnitude;
	}
};

template <typename K, int magnitude> 
__device__ __forceinline__ K MagnitudeShift(K key) {
	return MagnitudeShiftOp<K, (magnitude > 0) ? magnitude : magnitude * -1, (magnitude > 0)>::Shift(key);
}


//------------------------------------------------------------------------------
// SRTS Control Structures
//------------------------------------------------------------------------------


/**
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * cycle_elements greater than the normal, and the last workload 
 * does the extra (problem-size % cycle_elements) work.
 */
struct CtaDecomposition {
	unsigned int num_big_blocks;
	unsigned int big_block_elements;
	unsigned int normal_block_elements;
	unsigned int extra_elements_last_block;
};

template <typename K, typename V = K>
struct GlobalStorage {

	// Device vector of keys to sort
	K* keys;
	
	// Device vector of values to sort.  
	V* data;

	// Ancillary device vector for key storage 
	K* temp_keys;

	// Ancillary device vector for value storage
	V* temp_data;

	// Temporary device storage needed for radix sort.
	unsigned int *temp_spine;
};


//------------------------------------------------------------------------------
// SRTS Reduction & Scan/Scatter Configuration
//------------------------------------------------------------------------------

// 128 threads
#define SRTS_LOG_THREADS							7								
#define SRTS_THREADS								(1 << SRTS_LOG_THREADS)	

// Target threadblock occupancy for counting/reduction kernel
#define SM20_REDUCE_CTA_OCCUPANCY()					(8)			// 8 threadblocks on GF100
#define SM12_REDUCE_CTA_OCCUPANCY()					(5)			// 5 threadblocks on GT200
#define SM10_REDUCE_CTA_OCCUPANCY()					(4)			// 4 threadblocks on G80
#define SRTS_REDUCE_CTA_OCCUPANCY(version)			((version >= 200) ? SM20_REDUCE_CTA_OCCUPANCY() : 	\
													 (version >= 120) ? SM12_REDUCE_CTA_OCCUPANCY() : 	\
																		SM10_REDUCE_CTA_OCCUPANCY())		
													                    
// Target threadblock occupancy for bulk scan/scatter kernel
#define SM20_BULK_CTA_OCCUPANCY()					(7)			// 7 threadblocks on GF100
#define SM12_BULK_CTA_OCCUPANCY()					(5)			// 5 threadblocks on GT200
#define SM10_BULK_CTA_OCCUPANCY()					(2)			// 2 threadblocks on G80
#define SRTS_BULK_CTA_OCCUPANCY(version)			((version >= 200) ? SM20_BULK_CTA_OCCUPANCY() : 	\
													 (version >= 120) ? SM12_BULK_CTA_OCCUPANCY() : 	\
																		SM10_BULK_CTA_OCCUPANCY())		

// Number of 256-element sets to rake per raking pass
#define SM20_LOG_SETS_PER_PASS()					(1)			// 2 sets on GF100
#define SM12_LOG_SETS_PER_PASS()					(0)			// 1 set on GT200
#define SM10_LOG_SETS_PER_PASS()					(1)			// 2 sets on G80
#define SRTS_LOG_SETS_PER_PASS(version)				((version >= 200) ? SM20_LOG_SETS_PER_PASS() : 	\
													 (version >= 120) ? SM12_LOG_SETS_PER_PASS() : 	\
																		SM10_LOG_SETS_PER_PASS())		

// Number of raking passes per cycle
#define SM20_LOG_PASSES_PER_CYCLE(K, V)				(((MAX(sizeof(K), sizeof(V)) > 4) || SRTS_LP64) ? 0 : 1)	// 2 passes on GF100 (only one for large keys/values, or for 64-bit device pointers)
#define SM12_LOG_PASSES_PER_CYCLE(K, V)				(MAX(sizeof(K), sizeof(V)) > 4 ? 0 : 1)						// 2 passes on GT200 (only for large keys/values)
#define SM10_LOG_PASSES_PER_CYCLE(K, V)				(0)															// 1 pass on G80
#define SRTS_LOG_PASSES_PER_CYCLE(version, K, V)	((version >= 200) ? SM20_LOG_PASSES_PER_CYCLE(K, V) : 	\
													 (version >= 120) ? SM12_LOG_PASSES_PER_CYCLE(K, V) : 	\
																		SM10_LOG_PASSES_PER_CYCLE(K, V))		


// Number of raking threads per raking pass
#define SM20_LOG_RAKING_THREADS_PER_PASS()			(LOG_WARP_THREADS + 1)		// 2 raking warps on GF100
#define SM12_LOG_RAKING_THREADS_PER_PASS()			(LOG_WARP_THREADS)			// 1 raking warp on GT200
#define SM10_LOG_RAKING_THREADS_PER_PASS()			(LOG_WARP_THREADS + 2)		// 4 raking warps on G80
#define SRTS_LOG_RAKING_THREADS_PER_PASS(version)	((version >= 200) ? SM20_LOG_RAKING_THREADS_PER_PASS() : 	\
													 (version >= 120) ? SM12_LOG_RAKING_THREADS_PER_PASS() : 	\
																		SM10_LOG_RAKING_THREADS_PER_PASS())		

//
// Derived configuration parameters
//

// Number of elements per cycle
#define SRTS_LOG_CYCLE_ELEMENTS(version, K, V)		(SRTS_LOG_SETS_PER_PASS(version) + SRTS_LOG_PASSES_PER_CYCLE(version, K, V) + SRTS_LOG_THREADS + 1)
#define SRTS_CYCLE_ELEMENTS(version, K, V)			(1 << SRTS_LOG_CYCLE_ELEMENTS(version, K, V))

// Number of warps per CTA
#define SRTS_LOG_WARPS								(SRTS_LOG_THREADS - LOG_WARP_THREADS)
#define SRTS_WARPS									(1 << SRTS_LOG_WARPS)



//------------------------------------------------------------------------------
// Functors
//------------------------------------------------------------------------------

template <typename T>
struct NopFunctor{
	__device__ __host__ __forceinline__ void operator()(T &converted_key) {}
};

template <>
struct NopFunctor<char>{
	__device__ __host__ __forceinline__ void operator()(signed char &converted_key) {}		// Funny....
};

// Generic unsigned types

template <typename T> struct KeyConversion {
	typedef T UnsignedBits;
};

template <typename T>
struct PreprocessKeyFunctor{
	__device__ __host__ __forceinline__ void operator()(T &converted_key) {}
};

template <typename T>
struct PostprocessKeyFunctor {
	__device__ __host__ __forceinline__ void operator()(T &converted_key) {}
};


// Floats

template <> struct KeyConversion<float> {
	typedef unsigned int UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<float> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key) {
		int mask = -int(converted_key >> 31) | 0x80000000;
		converted_key ^= mask;
	}
};

template <>
struct PostprocessKeyFunctor<float> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key) {
        int mask = ((converted_key >> 31) - 1) | 0x80000000;
        converted_key ^= mask;
    }
};


// Doubles

template <> struct KeyConversion<double> {
	typedef unsigned long long UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<double> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key) {
		long long mask = -(long long)(converted_key >> 63) | 0x8000000000000000;
		converted_key ^= mask;
	}
};

template <>
struct PostprocessKeyFunctor<double> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key)  {
		long long mask = ((converted_key >> 63) - 1) | 0x8000000000000000;
        converted_key ^= mask;
    }
};


// Chars

template <> struct KeyConversion<char> {
	typedef unsigned char UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<char> {
	__device__ __host__ __forceinline__ void operator()(unsigned char &converted_key) {
		const unsigned int SIGN_MASK = 1u << ((sizeof(char) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
};

template <>
struct PostprocessKeyFunctor<char> {
	__device__ __host__ __forceinline__ void operator()(unsigned char &converted_key)  {
		const unsigned int SIGN_MASK = 1u << ((sizeof(char) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
};


// Shorts

template <> struct KeyConversion<short> {
	typedef unsigned short UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<short> {
	__device__ __host__ __forceinline__ void operator()(unsigned short &converted_key) {
		const unsigned int SIGN_MASK = 1u << ((sizeof(short) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
};

template <>
struct PostprocessKeyFunctor<short> {
	__device__ __host__ __forceinline__ void operator()(unsigned short &converted_key)  {
		const unsigned int SIGN_MASK = 1u << ((sizeof(short) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
};


// Ints

template <> struct KeyConversion<int> {
	typedef unsigned int UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<int> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key) {
		const unsigned int SIGN_MASK = 1u << ((sizeof(int) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
};

template <>
struct PostprocessKeyFunctor<int> {
	__device__ __host__ __forceinline__ void operator()(unsigned int &converted_key)  {
		const unsigned int SIGN_MASK = 1u << ((sizeof(int) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
};

// Longs

template <> struct KeyConversion<long> {
	typedef unsigned long UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<long> {
	__device__ __host__ __forceinline__ void operator()(unsigned long &converted_key) {
		const unsigned long SIGN_MASK = 1ul << ((sizeof(long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
};

template <>
struct PostprocessKeyFunctor<long> {
	__device__ __host__ __forceinline__ void operator()(unsigned long &converted_key)  {
		const unsigned long SIGN_MASK = 1ul << ((sizeof(long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
};


// LongLongs

template <> struct KeyConversion<long long> {
	typedef unsigned long long UnsignedBits;
};

template <>
struct PreprocessKeyFunctor<long long> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key) {
		const unsigned long long SIGN_MASK = 1ull << ((sizeof(long long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
	}
};

template <>
struct PostprocessKeyFunctor<long long> {
	__device__ __host__ __forceinline__ void operator()(unsigned long long &converted_key)  {
		const unsigned long long SIGN_MASK = 1ull << ((sizeof(long long) * 8) - 1);
		converted_key ^= SIGN_MASK;	
    }
};





//------------------------------------------------------------------------------
// Vector types
//------------------------------------------------------------------------------

template <typename K, int vec_elements> struct VecType;

// Arbitrary

template <typename K> 
struct VecType<K, 1> {
	typedef K Type;
};

template <typename K> 
struct VecType<K, 2> {
	K x;
	K y;
	typedef VecType<K, 2> Type;
};

template <typename K> 
struct VecType<K, 4> {
	K x;
	K y;
	K z;
	K w;
	typedef VecType<K, 4> Type;
};

// Floats

template<>
struct VecType<float, 1> {
	typedef float Type;
};

template<>
struct VecType<float, 2> {
	typedef float2 Type;
};

template<>
struct VecType<float, 4> {
	typedef float4 Type;
};

// Doubles

template<>
struct VecType<double, 1> {
	typedef double Type;
};

template<>
struct VecType<double, 2> {
	typedef double2 Type;
};

template<>
struct VecType<double, 4> {
	typedef double4 Type;
};

// Chars

template<>
struct VecType<char, 1> {
	typedef char Type;
};

template<>
struct VecType<char, 2> {
	typedef char2 Type;
};

template<>
struct VecType<char, 4> {
	typedef char4 Type;
};

// Unsigned chars

template<>
struct VecType<unsigned char, 1> {
	typedef unsigned char Type;
};

template<>
struct VecType<unsigned char, 2> {
	typedef uchar2 Type;
};

template<>
struct VecType<unsigned char, 4> {
	typedef uchar4 Type;
};

// Shorts

template<>
struct VecType<short, 1> {
	typedef short Type;
};

template<>
struct VecType<short, 2> {
	typedef short2 Type;
};

template<>
struct VecType<short, 4> {
	typedef short4 Type;
};

// Unsigned shorts

template<>
struct VecType<unsigned short, 1> {
	typedef unsigned short Type;
};

template<>
struct VecType<unsigned short, 2> {
	typedef ushort2 Type;
};

template<>
struct VecType<unsigned short, 4> {
	typedef ushort4 Type;
};

// Ints

template<>
struct VecType<int, 1> {
	typedef int Type;
};

template<>
struct VecType<int, 2> {
	typedef int2 Type;
};

template<>
struct VecType<int, 4> {
	typedef int4 Type;
};

// Unsigned ints

template<>
struct VecType<unsigned int, 1> {
	typedef unsigned int Type;
};

template<>
struct VecType<unsigned int, 2> {
	typedef uint2 Type;
};

template<>
struct VecType<unsigned int, 4> {
	typedef uint4 Type;
};

// Longs

template<>
struct VecType<long, 1> {
	typedef long Type;
};

template<>
struct VecType<long, 2> {
	typedef long2 Type;
};

template<>
struct VecType<long, 4> {
	typedef long4 Type;
};

// Unsigned longs

template<>
struct VecType<unsigned long, 1> {
	typedef unsigned long Type;
};

template<>
struct VecType<unsigned long, 2> {
	typedef ulong2 Type;
};

template<>
struct VecType<unsigned long, 4> {
	typedef ulong4 Type;
};

// Long longs

template<>
struct VecType<long long, 1> {
	typedef long long Type;
};

template<>
struct VecType<long long, 2> {
	typedef longlong2 Type;
};

template<>
struct VecType<long long, 4> {
	typedef longlong4 Type;
};

// Unsigned long longs

template<>
struct VecType<unsigned long long, 1> {
	typedef unsigned long long Type;
};

template<>
struct VecType<unsigned long long, 2> {
	typedef ulonglong2 Type;
};

template<>
struct VecType<unsigned long long, 4> {
	typedef ulonglong4 Type;
};



//------------------------------------------------------------------------------
// Common __forceinline__ Routines
//------------------------------------------------------------------------------


template <unsigned int NUM_ELEMENTS, bool MULTI_SCAN> 
__device__ __forceinline__ unsigned int WarpScan(
	volatile unsigned int warpscan[][NUM_ELEMENTS],
	unsigned int partial_reduction,
	unsigned int copy_section) {
	
	unsigned int warpscan_idx;
	if (MULTI_SCAN) {
		warpscan_idx = threadIdx.x & (NUM_ELEMENTS - 1);
	} else {
		warpscan_idx = threadIdx.x;
	}

	warpscan[1][warpscan_idx] = partial_reduction;

	if (NUM_ELEMENTS >= 2) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 1];
	if (NUM_ELEMENTS >= 4) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 2];
	if (NUM_ELEMENTS >= 8) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 4];
	if (NUM_ELEMENTS >= 16) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 8];
	if (NUM_ELEMENTS >= 32) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 16];
	
	if (copy_section > 0) {
		warpscan[1 + copy_section][warpscan_idx] = partial_reduction;
	}
	
	return warpscan[1][warpscan_idx - 1];
}


__device__ __forceinline__ void WarpReduce(
	unsigned int idx, 
	volatile unsigned int *smem_tree, 
	unsigned int partial_reduction) 
{
	smem_tree[idx] = partial_reduction;
	smem_tree[idx] = partial_reduction = partial_reduction + smem_tree[idx + 16];
	smem_tree[idx] = partial_reduction = partial_reduction + smem_tree[idx + 8];
	smem_tree[idx] = partial_reduction = partial_reduction + smem_tree[idx + 4];
	smem_tree[idx] = partial_reduction = partial_reduction + smem_tree[idx + 2];
	smem_tree[idx] = partial_reduction = partial_reduction + smem_tree[idx + 1];
}


__shared__ unsigned int vote_reduction[2][WARP_THREADS];
__device__ __forceinline__ unsigned int EmulatedWarpVoteAll(unsigned int predicate) {

	WarpReduce(threadIdx.x, vote_reduction[0], predicate);
	return (vote_reduction[0][0] == WARP_THREADS);
}


template <unsigned int LENGTH>
__device__ __forceinline__ unsigned int 
SerialReduce(unsigned int segment[]) {
	
	unsigned int reduce = segment[0];

	#pragma unroll
	for (int i = 1; i < (int) LENGTH; i++) {
		reduce += segment[i];
	}
	
	return reduce;
}


template <unsigned int LENGTH>
__device__ __forceinline__
void SerialScan(unsigned int segment[], unsigned int seed0) {
	
	unsigned int seed1;

	#pragma unroll	
	for (int i = 0; i < (int) LENGTH; i += 2) {
		seed1 = segment[i] + seed0;
		segment[i] = seed0;
		seed0 = seed1 + segment[i + 1];
		segment[i + 1] = seed1;
	}
}




//------------------------------------------------------------------------------
// Empty Kernels
//------------------------------------------------------------------------------

__global__ void FlushKernel()
{
}









#endif



