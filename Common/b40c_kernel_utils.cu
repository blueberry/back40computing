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
 * Statically determine log2(sizeof(T)), e.g., 
 * 		LogBytes<long long>::LOG_BYTES == 3
 * 		LogBytes<char[3]>::LOG_BYTES == 2
 */
template <typename T, int BYTES = sizeof(T), int LOG_VAL = 0>
struct LogBytes
{
	static const int LOG_BYTES = LogBytes<T, (BYTES >> 1), LOG_VAL + 1>::LOG_BYTES;
};

template <typename T, int LOG_VAL>
struct LogBytes<T, 0, LOG_VAL>
{
	static const int LOG_BYTES = (1 << (LOG_VAL - 1) < sizeof(T)) ? LOG_VAL : LOG_VAL - 1;
};


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
 * Work Management Datastructures 
 ******************************************************************************/

/**
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * subtile greater than the normal, and the last workload 
 * does the extra work.
 */
template <typename OffsetType>
struct CtaDecomposition {

	OffsetType num_elements;
	OffsetType total_subtiles;
	OffsetType subtiles_per_cta;
	OffsetType extra_subtiles;
	
	/**
	 * Constructor
	 */
	CtaDecomposition(OffsetType num_elements, int subtile_elements, int grid_size) :
		num_elements(num_elements),
		total_subtiles((num_elements + subtile_elements - 1) / subtile_elements),	// round up
		subtiles_per_cta(total_subtiles / grid_size),								// round down for the ks
		extra_subtiles(total_subtiles - (subtiles_per_cta * grid_size)) 			// the +1 subtilers
	{
	}
		
	/**
	 * Computes work limits for the current CTA
	 */	
	template <int LOG_TILE_ELEMENTS, int LOG_SUBTILE_ELEMENTS>
	__device__ __forceinline__ void GetCtaWorkLimits(
		OffsetType &cta_offset,				// Out param: Offset at which this CTA begins processing
		OffsetType &cta_elements,			// Out param: Total number of elements for this CTA to process
		OffsetType &guarded_offset, 		// Out param: Offset of final, partially-full tile (requires guarded loads)
		OffsetType &cta_guarded_elements)	// Out param: Number of elements in partially-full tile
	{
		const int TILE_ELEMENTS 		= 1 << LOG_TILE_ELEMENTS;
		const int SUBTILE_ELEMENTS 		= 1 << LOG_SUBTILE_ELEMENTS;
		
		// Compute number of elements and offset at which to start tile processing
		if (blockIdx.x < extra_subtiles) {
			// These CTAs get subtiles_per_cta+1 subtiles
			cta_elements = (subtiles_per_cta + 1) << LOG_SUBTILE_ELEMENTS;
			cta_offset = cta_elements * blockIdx.x;
		} else if (blockIdx.x < total_subtiles) {
			// These CTAs get subtiles_per_cta subtiles
			cta_elements = subtiles_per_cta << LOG_SUBTILE_ELEMENTS;
			cta_offset = (cta_elements * blockIdx.x) + (extra_subtiles << LOG_SUBTILE_ELEMENTS);
		} else {
			// These CTAs get no work (problem small enough that some CTAs don't even a single subtile)
			cta_elements = 0;
			cta_offset = 0;
		}
		
		// Compute (i) TILE aligned limit for tile-processing (oob), 
		// and (ii) how many extra guarded-load elements to process 
		// afterward (always less than a full tile) 
		if (cta_offset + cta_elements > num_elements) {
			// The last CTA having work will have rounded its last subtile up past the end 
			cta_elements = cta_elements - SUBTILE_ELEMENTS + 					// subtract subtile
				(num_elements & (SUBTILE_ELEMENTS - 1));		// add delta to end of input
		}
		cta_guarded_elements = cta_elements & (TILE_ELEMENTS - 1);				// The delta from the previous TILE alignment
		guarded_offset = cta_elements - cta_guarded_elements;
	}
};



/******************************************************************************
 * Common configuration types 
 ******************************************************************************/


/**
 * Description of a (typically) conflict-free serial-reduce-then-scan (SRTS) 
 * shared-memory grid 
 */
template <
	typename _PartialType,
	int _LOG_ACTIVE_THREADS, 
	int _LOG_PARTIALS_PER_THREAD,
	int _LOG_RAKING_THREADS> 
struct SrtsGrid
{
	typedef _PartialType							PartialType;
	
	enum {		// N.B.: We use an enum type here b/c of a NVCC-win compiler bug involving ternary expressions in static-const fields

		LOG_ACTIVE_THREADS				= _LOG_ACTIVE_THREADS,
		ACTIVE_THREADS					= 1 << LOG_ACTIVE_THREADS,

		LOG_PARTIALS_PER_THREAD			= _LOG_PARTIALS_PER_THREAD,
		PARTIALS_PER_THREAD				= 1 <<LOG_PARTIALS_PER_THREAD,

		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		LOG_PARTIALS					= LOG_ACTIVE_THREADS + LOG_PARTIALS_PER_THREAD,
		PARTIALS			 			= 1 << LOG_PARTIALS,

		LOG_PARTIALS_PER_SEG 			= LOG_PARTIALS - LOG_RAKING_THREADS,
		PARTIALS_PER_SEG 				= 1 << LOG_PARTIALS_PER_SEG,
	
		LOG_PARTIALS_PER_BANK_ARRAY		= B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__) +
											B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) -
											LogBytes<PartialType>::LOG_BYTES,
	
		LOG_PADDING_PARTIALS			= B40C_MAX(0, B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) - LogBytes<PartialType>::LOG_BYTES),
		PADDING_PARTIALS				= 1 << LOG_PADDING_PARTIALS,
	
		LOG_PARTIALS_PER_ROW			= B40C_MAX(LOG_PARTIALS_PER_SEG, LOG_PARTIALS_PER_BANK_ARRAY),
		PARTIALS_PER_ROW				= 1 << LOG_PARTIALS_PER_ROW,

		PADDED_PARTIALS_PER_ROW			= PARTIALS_PER_ROW + PADDING_PARTIALS,

		LOG_SEGS_PER_ROW 				= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG,
		SEGS_PER_ROW					= 1 << LOG_SEGS_PER_ROW,
	
		LOG_ROWS						= LOG_PARTIALS - LOG_PARTIALS_PER_ROW,
		ROWS 							= 1 << LOG_ROWS,
	
		PARTIAL_STRIDE_ROWS				= 1 << (LOG_ROWS - LOG_PARTIALS_PER_THREAD),
		PARTIAL_STRIDE					= PARTIAL_STRIDE_ROWS * PADDED_PARTIALS_PER_ROW,

		SMEM_BYTES						= ROWS * PADDED_PARTIALS_PER_ROW * sizeof(PartialType)
	};
	
	
	/**
	 * Returns the location in smem where the calling thread can insert/extract
	 * its partial for raking reduction/scan
	 */
	static __device__ __forceinline__ PartialType* BasePartial(PartialType *smem) 
	{
		int row = threadIdx.x >> LOG_PARTIALS_PER_ROW;		
		int col = threadIdx.x & (PARTIALS_PER_ROW - 1);			
		return smem + (row * PADDED_PARTIALS_PER_ROW) + col;
	}
	
	/**
	 * Returns the location in smem where the calling thread can begin serial 
	 * raking/scanning
	 */
	static __device__ __forceinline__ PartialType* RakingSegment(PartialType *smem) 
	{
		int row = threadIdx.x >> LOG_SEGS_PER_ROW;
		int col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		
		return smem + (row * PADDED_PARTIALS_PER_ROW) + col;
	}
};



/******************************************************************************
 * Common device routines (scans, reductions, etc.) 
 ******************************************************************************/


template <typename T, int NUM_ELEMENTS>
struct WarpScan
{
	// General iteration
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS], 
			int warpscan_tid) 
		{
			
			partial = partial + warpscan[1][warpscan_tid - OFFSET_LEFT];
			warpscan[1][warpscan_tid] = partial;
			Iterate<OFFSET_LEFT * 2>::Invoke(partial, warpscan, warpscan_tid);
		}
	};
	
	// Termination
	template <int __dummy>
	struct Iterate<NUM_ELEMENTS, __dummy>
	{
		static __device__ __forceinline__ void Invoke(
			T partial,
			volatile T warpscan[][NUM_ELEMENTS], 
			int warpscan_tid) {}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,								// Input partial
		T &total,								// Total aggregate reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],	// Smem for warpscanning
		int warpscan_tid = threadIdx.x)			// Thread's warpscan id 
	{
		warpscan[1][warpscan_tid] = partial;
		
		Iterate<1>::Invoke(partial, warpscan, warpscan_tid);

		// Set aggregate reduction
		total = warpscan[1][NUM_ELEMENTS - 1];
		
		// Return scan partial
		return warpscan[1][warpscan_tid - 1];
	}
};



/**
 * Perform a warp-synchronous reduction
 */
template <typename T, int NUM_ELEMENTS>
__device__ __forceinline__ void WarpReduce(
	int idx, 
	volatile T *storage, 
	T partial_reduction) 
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
struct SerialReduce
{
	// Iterate
	template <int COUNT, int TOTAL_COUNT>
	struct Iterate 
	{
		static __device__ __forceinline__ T Invoke(T partials[]) 
		{
			return partials[COUNT] + Iterate<COUNT + 1, TOTAL_COUNT>::Invoke(partials);
		}
	};
	
	// Terminate
	template <int TOTAL_COUNT>
	struct Iterate<TOTAL_COUNT, TOTAL_COUNT>
	{
		static __device__ __forceinline__ T Invoke(T partials[]) 
		{
			return partials[TOTAL_COUNT];
		}
	};
	
	// Interface
	static __device__ __forceinline__ T Invoke(T partials[])			
	{
		return Iterate<0, LENGTH - 1>::Invoke(partials);
	}
};



/**
 * Have each thread concurrently perform a serial scan over its 
 * specified segment (in place).  Returns the inclusive total.
 */
template <typename T, int LENGTH> 
struct SerialScan
{
	// Iterate
	template <int COUNT, int __dummy = 0>
	struct Iterate 
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial) 
		{
			T inclusive_partial = partials[COUNT] + exclusive_partial;
			results[COUNT] = exclusive_partial;
			return Iterate<COUNT + 1>::Invoke(partials, results, inclusive_partial);
		}
	};
	
	// Terminate
	template <int __dummy>
	struct Iterate<LENGTH, __dummy> 
	{
		static __device__ __forceinline__ T Invoke(T partials[], T results[], T exclusive_partial) 
		{
			return exclusive_partial;
		}
	};
	
	// Interface
	static __device__ __forceinline__ T Invoke(
		T partials[], 
		T exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0>::Invoke(partials, partials, exclusive_partial);
	}
	
	// Interface
	static __device__ __forceinline__ T Invoke(
		T partials[],
		T results[],
		T exclusive_partial)			// Exclusive partial to seed with
	{
		return Iterate<0>::Invoke(partials, results, exclusive_partial);
	}
};




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
	WarpReduce<int, ACTIVE_THREADS>(threadIdx.x, storage, predicate);
	return storage[0];
}


/**
 * Shared-memory reduction array for pre-Fermi voting 
 */
__shared__ int vote_reduction[B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];


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

