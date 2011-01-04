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
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 * Thanks!
 * 
 ******************************************************************************/

/******************************************************************************
 * Common B40C Routines 
 ******************************************************************************/

#pragma once

#include <b40c_cuda_properties.cu>
#include <b40c_kernel_data_movement.cu>

namespace b40c {


/******************************************************************************
 * Handy misc. routines  
 ******************************************************************************/

/**
 * Select maximum
 */
#define B40C_MAX(a, b) ((a > b) ? a : b)


/**
 * Select maximum
 */
#define B40C_MIN(a, b) ((a < b) ? a : b)


/**
 * Perform a swap
 */
template <typename T> 
void __host__ __device__ __forceinline__ Swap(T &a, T &b) {
	T temp = a;
	a = b;
	b = temp;
}


/**
 * MagnitudeShift().  Allows you to shift left for positive magnitude values, 
 * right for negative.   
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


/**
 * Supress warnings for unused constants
 */
template <typename T>
__device__ __forceinline__ void SuppressUnusedConstantWarning(const T) {}



/******************************************************************************
 * Common device routines (scans, reductions, etc.) 
 ******************************************************************************/

/**
 * Perform a warp-synchrounous prefix scan.  Allows for diverting a warp's
 * threads into separate scan problems (multi-scan). 
 */
template <int NUM_ELEMENTS, bool MULTI_SCAN> 
__device__ __forceinline__ int WarpScan(
	volatile int warpscan[][NUM_ELEMENTS],
	int partial_reduction,
	int copy_section = 0) {
	
	int warpscan_idx;
	if (MULTI_SCAN) {
		warpscan_idx = threadIdx.x & (NUM_ELEMENTS - 1);
	} else {
		warpscan_idx = threadIdx.x;
	}

	warpscan[1][warpscan_idx] = partial_reduction;

	if (NUM_ELEMENTS > 1) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 1];
	if (NUM_ELEMENTS > 2) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 2];
	if (NUM_ELEMENTS > 4) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 4];
	if (NUM_ELEMENTS > 8) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 8];
	if (NUM_ELEMENTS > 16) warpscan[1][warpscan_idx] = partial_reduction = 
			partial_reduction + warpscan[1][warpscan_idx - 16];
	
	if (copy_section > 0) {
		warpscan[1 + copy_section][warpscan_idx] = partial_reduction;
	}
	
	return warpscan[1][warpscan_idx - 1];
}


/**
 * Perform a warp-synchronous reduction
 */
template <int NUM_ELEMENTS>
__device__ __forceinline__ void WarpReduce(
	int idx, 
	volatile int *storage, 
	int partial_reduction) 
{
	storage[idx] = partial_reduction;

	if (NUM_ELEMENTS > 16) storage[idx] = partial_reduction = partial_reduction + storage[idx + 16];
	if (NUM_ELEMENTS > 8) storage[idx] = partial_reduction = partial_reduction + storage[idx + 8];
	if (NUM_ELEMENTS > 4) storage[idx] = partial_reduction = partial_reduction + storage[idx + 4];
	if (NUM_ELEMENTS > 2) storage[idx] = partial_reduction = partial_reduction + storage[idx + 2];
	if (NUM_ELEMENTS > 1) storage[idx] = partial_reduction = partial_reduction + storage[idx + 1];
}


/**
 * Have each thread concurrently perform a serial reduction over its specified segment 
 */
template <typename T, int LENGTH>
__device__ __forceinline__ 
T SerialReduce(T partials[]) {
	
	T reduce = partials[0];

	#pragma unroll
	for (int i = 1; i < LENGTH; i++) {
		reduce += partials[i];
	}
	
	return reduce;
}


/**
 * Have each thread concurrently perform a serial scan over its 
 * specified segment (in place).  Returns the inclusive total.
 */
template <typename T, int LENGTH>
__device__ __forceinline__
T SerialScan(T partials[], T seed) 
{
	// Unroll to avoid copy
	#pragma unroll	
	for (int i = 0; i <= LENGTH - 2; i += 2) {
		T tmp = partials[i] + seed;
		partials[i] = seed;
		seed = tmp + partials[i + 1];
		partials[i + 1] = tmp;
	}
	
	if (LENGTH & 1) {
		T tmp = partials[LENGTH - 1] + seed;
		partials[LENGTH - 1] = seed;
		seed = tmp;
	}
	
	return seed;
}


/**
 * Have each thread concurrently perform a serial scan over its specified segment
 */
template <typename T, int LENGTH>
__device__ __forceinline__
void SerialScan(
	T partials[], 
	T results[],
	T seed) 
{
	results[0] = seed;
	
	#pragma unroll	
	for (int i = 1; i < LENGTH; i++) {
		results[i] = results[i - 1] + partials[i - 1];
	}
}


/**
 * Simple wrapper for returning a CG-loaded int at the specified pointer  
 */
__device__ __forceinline__ int LoadCG(int* d_ptr) 
{
	int retval;
	ModifiedLoad<int, CG>::Ld(retval, d_ptr, 0);
	return retval;
}


/**
 * Implements a global, lock-free software barrier between gridDim.x CTAs using 
 * the synchronization vector d_sync having size gridDim.x 
 */
__device__ __forceinline__ void GlobalBarrier(int* d_sync) 
{
	// Threadfence and syncthreads to make sure global writes are visible before 
	// thread-0 reports in with its sync counter
	__threadfence();
	__syncthreads();
	
	if (blockIdx.x == 0) {

		// Report in ourselves
		if (threadIdx.x == 0) {
			d_sync[blockIdx.x] = 1; 
		}

		__syncthreads();
		
		// Wait for everyone else to report in
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
			while (LoadCG(d_sync + peer_block) == 0) {
				__threadfence_block();
			}
		}

		__syncthreads();
		
		// Let everyone know it's safe to read their prefix sums
		for (int peer_block = threadIdx.x; peer_block < gridDim.x; peer_block += blockDim.x) {
			d_sync[peer_block] = 0;
		}

	} else {
		
		if (threadIdx.x == 0) {
			// Report in 
			d_sync[blockIdx.x] = 1; 

			// Wait for acknowledgement
			while (LoadCG(d_sync + blockIdx.x) == 1) {
				__threadfence_block();
			}
		}
		
		__syncthreads();
	}
}


/******************************************************************************
 * Common device intrinsics specialized by architecture  
 ******************************************************************************/

/**
 * Terminates the calling thread
 */
__device__ __forceinline__ static void ThreadExit() {				
	asm("exit;");
}	


/**
 * The best way to multiply integers (24 effective bits or less)
 */
#if __CUDA_ARCH__ >= 200
	#define FastMul(a, b) (a * b)
#else
	#define FastMul(a, b) (__umul24(a, b))
#endif	

/**
 * The best way to warp-vote
 */
#if __CUDA_ARCH__ >= 120
	#define WarpVoteAll(active_threads, predicate) (__all(predicate))
#else 
	#define WarpVoteAll(active_threads, predicate) (EmulatedWarpVoteAll<active_threads>(predicate))
#endif

#if __CUDA_ARCH__ >= 200
	#define TallyWarpVote(active_threads, predicate, storage) (__popc(__ballot(predicate)))
#else 
	#define TallyWarpVote(active_threads, predicate, storage) (TallyWarpVoteSm10<active_threads>(predicate, storage))
#endif


/**
 * Tally a warp-vote regarding the given predicate using the supplied storage
 */
template <int ACTIVE_THREADS>
__device__ __forceinline__ int TallyWarpVoteSm10(int predicate, int storage[]) {
	WarpReduce<ACTIVE_THREADS>(threadIdx.x, storage, predicate);
	return storage[0];
}


/**
 * Shared-memory reduction array for pre-Fermi voting 
 */
__shared__ int vote_reduction[B40C_WARP_THREADS];


/**
 * Tally a warp-vote regarding the given predicate
 */
template <int ACTIVE_THREADS>
__device__ __forceinline__ int TallyWarpVoteSm10(int predicate) {
	return TallyWarpVoteSm10<ACTIVE_THREADS>(predicate, vote_reduction);
}


/**
 * Emulate the __all() warp vote instruction
 */
template <int ACTIVE_THREADS>
__device__ __forceinline__ int EmulatedWarpVoteAll(int predicate) {
	return (TallyWarpVoteSm10<ACTIVE_THREADS>(predicate) == ACTIVE_THREADS);
}



/******************************************************************************
 * Simple Kernels
 ******************************************************************************/

/**
 * Zero's out a vector. 
 */
template <typename T>
__global__ void MemsetKernel(T *d_out, T value, int length)
{
	int STRIDE = gridDim.x * blockDim.x;
	for (int idx = (blockIdx.x * blockDim.x) + threadIdx.x; idx < length; idx += STRIDE) {
		d_out[idx] = value;
	}
}


} // namespace b40c

