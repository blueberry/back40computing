/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * Cooperative tile reduction and scanning within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/raking_grid.cuh>
#include <b40c/util/numeric_traits.cuh>
#include <b40c/util/reduction/cooperative_reduction.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>

namespace b40c {
namespace util {
namespace scan {


/**
 *
 */
template <
	int CUDA_ARCH,						// CUDA SM architecture to generate code for
	int LOG_THREADS,
	int LOG_LOAD_VEC_SIZE,
	int LOG_LOADS_PER_TILE,
	typename T,							// Data type of scan partials
	typename ReductionOp,				// Binary associative reduction functor for pairs of elements of type T
	typename IdentityOp = NullType>		// An associative identity functor for the scan operation vastly improves performance.  (The identity_op may be an instance of NullType if no such identity exists)
struct CooperativeTileScan
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	// Warpscan type
	typedef WarpScan<
		T,
		ReductionOp,
		IdentityOp> WarpScan;

	// Constants
	enum {
		LOG_RAKING_THREADS		= Warpscan::LOG_WARPSCAN_THREADS,
		RAKING_THREADS			= 1 << LOG_RAKING_THREADS,

		LOAD_VEC_SIZE 			= 1 << LOG_LOAD_VEC_SIZE,
		LOADS_PER_TILE			= 1 << LOG_LOADS_PER_TILE,

		LOG_SCAN_LANES			= LOG_LOADS_PER_TILE,
		SCAN_LANES				= 1 << LOG_SCAN_LANES,

		HAS_IDENTITY			= Equals<IdentityOp, NullType>::NEGATE,
	};

	// Raking grid type (having LOADS_PER_TILE lanes)
	typedef RakingGrid<
		CUDA_ARCH,
		T,
		LOG_THREADS,
		LOG_RAKING_THREADS,
		LOG_SCAN_LANES> RakingGrid;



	//---------------------------------------------------------------------
	// Opaque shared storage types needed to construct CooperativeTileScan
	//---------------------------------------------------------------------

	// Warpscan storage type.  Should not be re-purposed.
	typedef typename WarpScan::WarpscanStorage WarpscanStorage;

	// Lane storage type.  Can be re-purposed.
	typedef typename RakingGrid::LaneStorage LaneStorage;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	WarpScan				warp_scan;				// Warpscan utility
	RakingGrid 				raking_grid;			// Raking grid utility

	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Iterate over loads (next load)
	 */
	template <int LOAD, int TOTAL_LOADS>
	struct IterateLoad
	{
		// Serial upsweep reduction
		static __device__ __forceinline__ void UpsweepReduction(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			ReductionOp scan_op,
			CooperativeTileScan *tile_scan)
		{
			// Reduce the partials in this lane/load
			T partial_reduction = SerialReduce<LOAD_VEC_SIZE>::Invoke(
				data[LOAD],
				reduction_op);

			// Store partial reduction into SRTS grid
			tile_scan->raking_grid.my_lane_partial[LOAD][0] = partial_reduction;

			// Next load
			IterateLoad<LOAD + 1, TOTAL_LOADS>::UpsweepReduction(
				srts_details,
				data,
				reduction_op);
		}

		// Sequential downsweep scan
		static __device__ __forceinline__ void DownsweepScan(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			ReductionOp scan_op,
			CooperativeTileScan *tile_scan)
		{
			// Retrieve partial reduction from raking grid
			T exclusive_partial = tile_scan->raking_grid.my_lane_partial[LOAD][0];

			// Serial scan the partials in this lane/load
			SerialScan<LOAD_VEC_SIZE, EXCLUSIVE_SCAN>::Invoke(
				data[LOAD],
				exclusive_partial,
				scan_op);

			// Next load
			IterateLoad<LOAD + 1, TOTAL_LOADS>::DownsweepScan(
				srts_details, data, scan_op);
		}
	};


	/**
	 * Iterate over loads (terminate)
	 */
	template <int TOTAL_LOADS>
	struct IterateLoad<TOTAL_LOADS, TOTAL_LOADS>
	{
		// Serial upsweep reduction
		static __device__ __forceinline__ void UpsweepReduction(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			ReductionOp scan_op,
			CooperativeTileScan *tile_scan) {}

		// Sequential downsweep scan
		static __device__ __forceinline__ void DownsweepScan(
			T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
			ReductionOp scan_op,
			CooperativeTileScan *tile_scan) {}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor.
	 *
	 * Specifying an associative identity functor for the scan operation vastly
	 * improves performance.  (The identity_op may be an instance of NullType
	 * if no such identity exists)
	 */
	__device__ __forceinline__ CooperativeTileScan(
		WarpscanStorage 	&warpscan_storage,
		LaneStorage 		&lane_storage,
		ReductionOp 		reduction_op,
		IdentityOp 			identity_op = NullType()) :
			// Initializers
			warpscan_storage(warpscan_storage),
			raking_grid(lane_storage),
			reduction_op(reduction_op),
			identity_op(identity_op)
	{
		InitWarpscanStorage<IdentityOp, RAKING_THREADS>::Init(
			warpscan_storage,
			identity_op);
	}


	/**
	 * Inclusive prefix scan a CTA tile of data.
	 * Returns the total aggregate in all threads.
	 *
	 * No synchronization needed before storage reuse.
	 */
	__device__ __forceinline__ T InclusivePrefixScan(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE])
	{
		// Reduce partials in each load, placing resulting partials in raking grid lanes
		IterateLoad<0, SCAN_LANES>::UpsweepReduction(data, scan_op, this);

		__syncthreads();

		if (threadIdx.x < RAKING_THREADS) {

			// Inclusive raking scan
			SerialScan<RakingGrid::PARTIALS_PER_SEG, false>::Invoke(
				raking_grid.my_raking_segment,
				exclusive_partial,
				scan_op);

			// Exclusive warp scan
			T exclusive_partial = WarpScan<LOG_RAKING_THREADS>::Invoke(
				inclusive_partial,
				warpscan_storage,
				scan_op);

		}

		__syncthreads();

		// Scan each load, seeded with the resulting partial its raking lane
		IterateLoad<0, SCAN_LANES>::DownsweepScan(data, scan_op, this);

		// Return total aggregate from warpscan
		return warpscan_storage[1][RAKING_THREADS - 1];
	}


	/**
	 * Prefix scan a CTA tile of data.  Returns the total aggregate in all threads.
	 *
	 * No synchronization needed before storage reuse.
	 */
	template <typename ReductionOp>
	__device__ __forceinline__ T PrefixScan(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
		ReductionOp scan_op)
	{
		// Reduce partials in each load, placing resulting partials in raking grid lanes
		IterateLoad<0, SCAN_LANES>::UpsweepReduction(data, scan_op, this);

		__syncthreads();

		if (threadIdx.x < RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingGrid::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				scan_op);

			// Exclusive warp scan
			T exclusive_partial = WarpScan<LOG_RAKING_THREADS>::Invoke(
				inclusive_partial,
				warpscan_storage,
				scan_op);

			// Exclusive raking scan
			SerialScan<RakingGrid::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				exclusive_partial,
				scan_op);
		}

		__syncthreads();

		// Scan each load, seeded with the resulting partial its raking lane
		IterateLoad<0, SCAN_LANES>::DownsweepScan(data, scan_op, this);

		// Return total aggregate from warpscan
		return warpscan_storage[1][RAKING_THREADS - 1];
	}


	/**
	 * Prefix scan CTA tile of data.  Scanned data is seeded with carry, and
	 * carry is subsequently updated with the total aggregate.  Carry is
	 * valid only in raking threads.
	 *
	 * No synchronization needed before storage reuse.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void PrefixScanWithCarry(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
		T &carry,
		ReductionOp scan_op)
	{
		// Reduce partials in each load, placing resulting partials in raking grid lanes
		IterateLoad<0, SCAN_LANES>::UpsweepReduction(data, scan_op, this);

		__syncthreads();

		if (threadIdx.x < RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingGrid::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<LOG_RAKING_THREADS>::Invoke(
				inclusive_partial,
				warpscan_total,
				warpscan_storage,
				scan_op);

			// Seed exclusive partial with carry-in
			exclusive_partial = scan_op(carry, exclusive_partial);

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				exclusive_partial,
				scan_op);

			// Update carry
			carry = scan_op(carry, warpscan_total);
		}

		__syncthreads();

		// Scan each load, seeded with the resulting partial its raking lane
		IterateLoad<0, SCAN_LANES>::DownsweepScan(data, scan_op, this);
	}


	/**
	 * Prefix sum CTA tile of data.  The specified queue counter is atomically
	 * incremented by the total aggregate.  Scanned data is seeded with
	 * the counter's previous value.  Returns the total aggregate
	 * (including previous queue counter) in all threads.
	 *
	 * T must be a built-in integer type.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	__device__ __forceinline__ T PrefixSumWithEnqueue(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
		T* d_enqueue_counter)
	{
		util::Sum<T> scan_op;

		// Reduce partials in each load, placing resulting partials in raking grid lanes
		IterateLoad<0, SCAN_LANES>::UpsweepReduction(data, scan_op, this);

		__syncthreads();

		if (threadIdx.x < RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<RakingGrid::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<LOG_RAKING_THREADS>::Invoke(
				inclusive_partial,
				warpscan_total,
				warpscan_storage,
				scan_op);

			// Atomic-increment the global counter with the total allocation
			T reservation_offset;
			if (threadIdx.x == 0) {
				reservation_offset = AtomicInt<T>::Add(d_enqueue_counter, warpscan_total);

				// Share previous counter
				warpscan_storage[1][0] = reservation_offset;

				// Share updated counter
				warpscan_storage[1][RAKING_THREADS] =  scan_op(reservation_offset, warpscan_total);
			}

			// Seed exclusive partial with queue reservation offset
			reservation_offset = warpscan_storage[1][0];
			exclusive_partial = scan_op(reservation_offset, exclusive_partial);

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				exclusive_partial,
				scan_op);
		}

		__syncthreads();

		// Scan each load, seeded with the resulting partial its raking lane
		IterateLoad<0, SCAN_LANES>::DownsweepScan(data, scan_op, this);

		// Return updated queue counter
		return warpscan_storage[1][RAKING_THREADS];
	}


	/**
	 * Prefix sum CTA tile of data.  The specified queue counter is atomically
	 * incremented by the total aggregate, and the previous queue counter
	 * is returned in enqueue_offset in all threads.  Scanned data is *not*
	 * seeded with the counter's previous value.  Returns the total aggregate
	 * (*not* including previous queue counter) in all threads.
	 *
	 * T must be a built-in integer type.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	__device__ __forceinline__ T PrefixSumWithEnqueue(
		T data[LOADS_PER_TILE][LOAD_VEC_SIZE],
		T *d_enqueue_counter,
		T &enqueue_offset)
	{
		util::Sum<T> scan_op;

		// Reduce partials in each load, placing resulting partials in raking grid lanes
		IterateLoad<0, SCAN_LANES>::UpsweepReduction(data, scan_op, this);

		__syncthreads();

		if (threadIdx.x < RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				srts_details.raking_segment, scan_op);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<SrtsDetails::LOG_RAKING_THREADS>::Invoke(
				inclusive_partial, warpscan_total, srts_details.warpscan, scan_op);

			// Atomic-increment the global counter with the total allocation
			T reservation_offset;
			if (threadIdx.x == 0) {
				reservation_offset = AtomicInt<T>::Add(d_enqueue_counter, warpscan_total);

				// Share previous counter
				warpscan_storage[1][0] = reservation_offset;
			}

			// Exclusive raking scan
			SerialScan<SrtsDetails::PARTIALS_PER_SEG>::Invoke(
				raking_grid.my_raking_segment,
				exclusive_partial,
				scan_op);
		}

		__syncthreads();

		// Scan each load, seeded with the resulting partial its raking lane
		IterateLoad<0, SCAN_LANES>::DownsweepScan(data, scan_op, this);

		// Retrieve previous queue counter
		enqueue_offset = warpscan_storage[1][0];

		// Return total aggregate from warpscan
		return warpscan_storage[1][RAKING_THREADS - 1];
	}

};







} // namespace scan
} // namespace util
} // namespace b40c

