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

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"

namespace b40c {


/******************************************************************************
 * Handy misc. routines  
 ******************************************************************************/

/**
 * Displays error message in accordance with debug mode
 */
cudaError_t B40CPerror(cudaError_t error, const char *message, const char *filename, int line)
{
	if (error) {
		fprintf(stderr, "[%s, %d] %s (CUDA error %d: %s)\n", filename, line, message, error, cudaGetErrorString(error));
		fflush(stderr);
	}
	return error;
}


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
	// Inductive case
	static const int LOG_BYTES = LogBytes<T, (BYTES >> 1), LOG_VAL + 1>::LOG_BYTES;
};

template <typename T, int LOG_VAL>
struct LogBytes<T, 0, LOG_VAL>
{
	// Base case
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


/******************************************************************************
 * Work Management Datastructures 
 ******************************************************************************/

/**
 * Description of work distribution amongst CTAs
 *
 * A given threadblock may receive one of three different amounts of 
 * work: "big", "normal", and "last".  The big workloads are one
 * grain greater than the normal, and the last workload 
 * does the extra work.
 */
template <typename SizeT> 		// Integer type for indexing into problem arrays (e.g., int, long long, etc.)
struct CtaWorkDistribution
{
	SizeT num_elements;		// Number of elements in the problem
	SizeT total_grains;		// Number of "grain" blocks to break the problem into (round up)
	SizeT grains_per_cta;	// Number of "grain" blocks per CTA
	SizeT extra_grains;		// Number of CTAs having one extra "grain block"
	int grid_size;			// Number of CTAs

	/**
	 * Constructor
	 */
	CtaWorkDistribution(
		SizeT num_elements,
		int schedule_granularity, 	// Problem granularity by which work is distributed amongst CTA threadblocks
		int grid_size) :
			num_elements(num_elements),
			total_grains((num_elements + schedule_granularity - 1) / schedule_granularity),		// round up
			grains_per_cta((grid_size > 0) ? total_grains / grid_size : 0),						// round down for the ks
			extra_grains(total_grains - (grains_per_cta * grid_size)), 							// the CTAs with +1 grains
			grid_size(grid_size)
	{}


	/**
	 * Computes work limits for the current CTA
	 */	
	template <
		int LOG_TILE_ELEMENTS,			// CTA tile size, i.e., granularity by which the CTA processes work
		int LOG_SCHEDULE_GRANULARITY>	// Problem granularity by which work is distributed amongst CTA threadblocks
	__device__ __forceinline__ void GetCtaWorkLimits(
		SizeT &cta_offset,				// Out param: Offset at which this CTA begins processing
		SizeT &cta_elements,			// Out param: Total number of elements for this CTA to process
		SizeT &guarded_offset, 			// Out param: Offset of final, partially-full tile (requires guarded loads)
		SizeT &cta_guarded_elements)	// Out param: Number of elements in partially-full tile
	{
		const int TILE_ELEMENTS 				= 1 << LOG_TILE_ELEMENTS;
		const int SCHEDULE_GRANULARITY 			= 1 << LOG_SCHEDULE_GRANULARITY;
		
		// Compute number of elements and offset at which to start tile processing
		if (blockIdx.x < extra_grains) {
			// This CTA gets grains_per_cta+1 grains
			cta_elements = (grains_per_cta + 1) << LOG_SCHEDULE_GRANULARITY;
			cta_offset = cta_elements * blockIdx.x;

		} else if (blockIdx.x < total_grains) {
			// This CTA gets grains_per_cta grains
			cta_elements = grains_per_cta << LOG_SCHEDULE_GRANULARITY;
			cta_offset = (cta_elements * blockIdx.x) + (extra_grains << LOG_SCHEDULE_GRANULARITY);

		} else {
			// This CTA gets no work (problem small enough that some CTAs don't even a single grain)
			cta_elements = 0;
			cta_offset = 0;
		}
		
		if (cta_offset + cta_elements > num_elements) {
			// The last CTA having work will have rounded its last grain up past the end 
			cta_elements = cta_elements - SCHEDULE_GRANULARITY + 			// subtract grain size
				(num_elements & (SCHEDULE_GRANULARITY - 1));				// add delta to end of input
		}

		// The tile-aligned limit for full-tile processing
		cta_guarded_elements = cta_elements & (TILE_ELEMENTS - 1);

		// The number of extra guarded-load elements to process afterward (always
		// less than a full tile)
		guarded_offset = cta_offset + cta_elements - cta_guarded_elements;
	}
};



/******************************************************************************
 * Common configuration types 
 ******************************************************************************/


/**
 * Description of a (typically) conflict-free serial-reduce-then-scan (SRTS) 
 * shared-memory grid.
 *
 * A "lane" for reduction/scan consists of one value (i.e., "partial") per
 * active thread.  A grid consists of one or more scan lanes. The lane(s)
 * can be sequentially "raked" by the specified number of raking threads
 * (e.g., for upsweep reduction or downsweep scanning), where each raking
 * thread progresses serially through a segment that is its share of the
 * total grid.
 *
 * Depending on how the raking threads are further reduced/scanned, the lanes
 * can be independent (i.e., only reducing the results from every
 * SEGS_PER_LANE raking threads), or fully dependent (i.e., reducing the
 * results from every raking thread)
 */
template <
	typename _PartialType,		// Type of items we will be reducing/scanning
	int _LOG_ACTIVE_THREADS, 	// Number of threads placing a lane partial (i.e., the number of partials per lane)
	int _LOG_SCAN_LANES,		// Number of independent scan lanes
	int _LOG_RAKING_THREADS> 	// Number of threads used for raking (typically 1 warp)
struct SrtsGrid
{
	// Type of items we will be reducing/scanning
	typedef _PartialType				PartialType;
	
	// N.B.: We use an enum type here b/c of a NVCC-win compiler bug where the
	// compiler can't handle ternary expressions in static-const fields having
	// both evaluation targets as local const expressions.
	enum {

		// Number number of partials per lane
		LOG_PARTIALS_PER_LANE 			= _LOG_ACTIVE_THREADS,
		PARTIALS_PER_LANE				= 1 << LOG_PARTIALS_PER_LANE,

		// Number of scan lanes
		LOG_SCAN_LANES					= _LOG_SCAN_LANES,
		SCAN_LANES						= 1 <<LOG_SCAN_LANES,

		// Number of raking threads
		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		// Total number of partials in the grid (all lanes)
		LOG_PARTIALS					= LOG_PARTIALS_PER_LANE + LOG_SCAN_LANES,
		PARTIALS			 			= 1 << LOG_PARTIALS,

		// Partials to be raked per raking thread
		LOG_PARTIALS_PER_SEG 			= LOG_PARTIALS - LOG_RAKING_THREADS,
		PARTIALS_PER_SEG 				= 1 << LOG_PARTIALS_PER_SEG,
	
		// Number of partials that we can put in one stripe across the shared memory banks
		LOG_PARTIALS_PER_BANK_ARRAY		= B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__) +
											B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) -
											LogBytes<PartialType>::LOG_BYTES,
	
		// Number of partials that we must use to "pad out" one memory bank
		LOG_PADDING_PARTIALS			= B40C_MAX(0, B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) - LogBytes<PartialType>::LOG_BYTES),
		PADDING_PARTIALS				= 1 << LOG_PADDING_PARTIALS,
	
		// Number of consecutive partials we can have without padding (i.e., a "row")
		LOG_PARTIALS_PER_ROW			= B40C_MAX(LOG_PARTIALS_PER_SEG, LOG_PARTIALS_PER_BANK_ARRAY),
		PARTIALS_PER_ROW				= 1 << LOG_PARTIALS_PER_ROW,

		// Number of partials (including padding) per "row"
		PADDED_PARTIALS_PER_ROW			= PARTIALS_PER_ROW + PADDING_PARTIALS,

		// Number of raking segments per row (i.e., number of raking threads per row)
		LOG_SEGS_PER_ROW 				= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG,
		SEGS_PER_ROW					= 1 << LOG_SEGS_PER_ROW,
	
		// Number of rows in the grid
		LOG_ROWS						= LOG_PARTIALS - LOG_PARTIALS_PER_ROW,
		ROWS 							= 1 << LOG_ROWS,
	
		// Number of rows per lane
		LOG_ROWS_PER_LANE				= LOG_ROWS - LOG_SCAN_LANES,
		ROWS_PER_LANE					= 1 << LOG_ROWS_PER_LANE,

		// Number of raking thraeds per lane
		LOG_RAKING_THREADS_PER_LANE		= LOG_SEGS_PER_ROW + LOG_ROWS_PER_LANE,
		RAKING_THREADS_PER_LANE			= 1 << LOG_RAKING_THREADS_PER_LANE,

		// Stride between lanes (in partials)
		LANE_STRIDE						= ROWS_PER_LANE * PADDED_PARTIALS_PER_ROW,

		// Total number of smem bytes needed to back the grid
		SMEM_BYTES						= ROWS * PADDED_PARTIALS_PER_ROW * sizeof(PartialType)
	};
	
	
	/**
	 * Returns the location in the smem grid where the calling thread can insert/extract
	 * its partial for raking reduction/scan into the first lane.  Positions in subsequent
	 * lanes can be obtained via increments of LANE_STRIDE.
	 */
	static __device__ __forceinline__ PartialType* BasePartial(PartialType *smem) 
	{
		int row = threadIdx.x >> LOG_PARTIALS_PER_ROW;		
		int col = threadIdx.x & (PARTIALS_PER_ROW - 1);			
		return smem + (row * PADDED_PARTIALS_PER_ROW) + col;
	}
	
	/**
	 * Returns the location in the smem grid where the calling thread can begin serial
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

/**
 * Binary associative operators
 */
namespace binary_ops
{

template <typename T>
T __host__ __device__ __forceinline__ Sum(const T &a, const T &b)
{
	return a + b;
}

template <typename T>
T __host__ __device__ __forceinline__ Max(const T &a, const T &b)
{
	return (a > b) ? a : b;
}

} // namespace binary_ops


/**
 * Performs NUM_ELEMENTS steps of a Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 */
template <typename T, int LOG_NUM_ELEMENTS, int STEPS = LOG_NUM_ELEMENTS>
struct WarpScanInclusive
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;
	static const int WIDTH = 1 << STEPS;

	// General iteration
	template <int OFFSET_LEFT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ int Invoke(
			T partial, volatile T ks_warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			ks_warpscan[1][warpscan_tid] = partial;
			partial = partial + ks_warpscan[1][warpscan_tid - OFFSET_LEFT];
			return Iterate<OFFSET_LEFT * 2>::Invoke(partial, ks_warpscan, warpscan_tid);
		}
	};

	// Termination
	template <int __dummy>
	struct Iterate<WIDTH, __dummy>
	{
		static __device__ __forceinline__ int Invoke(
			T partial, volatile T ks_warpscan[][NUM_ELEMENTS], int warpscan_tid)
		{
			return partial;
		}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,									// Input partial
		volatile T ks_warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
	{
		return Iterate<1>::Invoke(partial, ks_warpscan, warpscan_tid);
	}
};


/**
 * Performs NUM_ELEMENTS steps of a Kogge-Stone style prefix scan.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * Can be used to perform concurrent, independent warp-scans if
 * storage pointers and their local-thread indexing id's are set up properly.
 */
template <typename T, int LOG_NUM_ELEMENTS>
struct WarpScan
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

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
		T partial,									// Input partial
		T &total,									// Total aggregate reduction (out param)
		volatile T warpscan[][NUM_ELEMENTS],		// Smem for warpscanning containing at least two segments of size NUM_ELEMENTS (the first being initialized to zero's)
		int warpscan_tid = threadIdx.x)				// Thread's local index into a segment of NUM_ELEMENTS items
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
 * Perform a warp-synchronous reduction.
 *
 * This procedure assumes that no explicit barrier synchronization is needed
 * between steps (i.e., warp-synchronous programming)
 *
 * Can be used to perform concurrent, independent warp-reductions if
 * storage pointers and their local-thread indexing id's are set up properly.
 */
template <
	typename T,
	int LOG_NUM_ELEMENTS,
	T BinaryOp(const T&, const T&) = binary_ops::Sum>
struct WarpReduce
{
	static const int NUM_ELEMENTS = 1 << LOG_NUM_ELEMENTS;

	// General iteration
	template <int OFFSET_RIGHT, int __dummy = 0>
	struct Iterate
	{
		static __device__ __forceinline__ void Invoke(T partial, volatile T *storage, int tid) 
		{
			T from_storage = storage[tid + OFFSET_RIGHT];
			partial = BinaryOp(partial, from_storage);
			storage[tid] = partial;
			Iterate<OFFSET_RIGHT / 2>::Invoke(partial, storage, tid);
		}
	};
	
	// Termination
	template <int __dummy>
	struct Iterate<0, __dummy>
	{
		static __device__ __forceinline__ void Invoke(T partial, volatile T *storage, int tid) {}
	};

	// Interface
	static __device__ __forceinline__ T Invoke(
		T partial,					// Input partial
		volatile T *storage,		// Smem for reducing of length equal to at least 1.5x NUM_ELEMENTS
		int tid = threadIdx.x)		// Thread's local index into a segment of NUM_ELEMENTS items
	{
		storage[tid] = partial;
		Iterate<NUM_ELEMENTS / 2>::Invoke(partial, storage, tid);
		return storage[0];
	}
};



/**
 * Have each thread concurrently perform a serial reduction over its specified segment 
 */
template <
	typename T,
	int LENGTH,
	T BinaryOp(const T&, const T&) = binary_ops::Sum>
struct SerialReduce
{
	// Iterate
	template <int COUNT, int TOTAL>
	struct Iterate 
	{
		static __device__ __forceinline__ T Invoke(T partials[]) 
		{
			T a = Iterate<COUNT - 2, TOTAL>::Invoke(partials);
			T b = partials[TOTAL - COUNT];
			T c = partials[TOTAL - (COUNT - 1)];

			// TODO: consider specializing with a video 3-op instructions on SM2.0+, e.g., asm("vadd.s32.s32.s32.add %0, %1, %2, %3;" : "=r"(a) : "r"(a), "r"(b), "r"(c));
			return BinaryOp(a, BinaryOp(b, c));
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<2, TOTAL>
	{
		static __device__ __forceinline__ T Invoke(T partials[])
		{
			return BinaryOp(partials[TOTAL - 2], partials[TOTAL - 1]);
		}
	};

	// Terminate
	template <int TOTAL>
	struct Iterate<1, TOTAL>
	{
		static __device__ __forceinline__ T Invoke(T partials[]) 
		{
			return partials[TOTAL - 1];
		}
	};
	
	// Interface
	static __device__ __forceinline__ T Invoke(T partials[])			
	{
		return Iterate<LENGTH, LENGTH>::Invoke(partials);
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
 * The best way to warp-vote (for the first warp only)
 */
#if __CUDA_ARCH__ >= 120
	#define WarpVoteAll(log_active_threads, predicate) (__all(predicate))
#else 
	#define WarpVoteAll(log_active_threads, predicate) (EmulatedWarpVoteAll<log_active_threads>(predicate))
#endif

#if __CUDA_ARCH__ >= 200
	#define TallyWarpVote(log_active_threads, predicate, storage) (__popc(__ballot(predicate)))
#else 
	#define TallyWarpVote(log_active_threads, predicate, storage) (TallyWarpVoteSm10<log_active_threads>(predicate, storage))
#endif


/**
 * Tally a warp-vote regarding the given predicate using the supplied storage
 * (For the first warp only)
 */
template <int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int TallyWarpVoteSm10(int predicate, int storage[]) {
	return WarpReduce<int, LOG_ACTIVE_THREADS>::Invoke(predicate, storage);
}


/**
 * Tally a warp-vote regarding the given predicate
 * (For the first warp only)
 */
__shared__ int vote_reduction[B40C_WARP_THREADS(__B40C_CUDA_ARCH__) * 2];
template <int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int TallyWarpVoteSm10(int predicate) {
	return TallyWarpVoteSm10<LOG_ACTIVE_THREADS>(predicate, vote_reduction);
}


/**
 * Emulate the __all() warp vote instruction
 * (For the first warp only)
 */
template <int LOG_ACTIVE_THREADS>
__device__ __forceinline__ int EmulatedWarpVoteAll(int predicate) {
	const int ACTIVE_THREADS = 1 << LOG_ACTIVE_THREADS;
	return (TallyWarpVoteSm10<LOG_ACTIVE_THREADS>(predicate) == ACTIVE_THREADS);
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

