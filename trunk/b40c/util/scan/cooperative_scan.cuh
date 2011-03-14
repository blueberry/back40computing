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

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/cooperative_reduction.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_scan.cuh>

namespace b40c {
namespace util {
namespace scan {

/**
 * Cooperative reduction in SRTS grid hierarchies
 */
template <
	typename SrtsDetails,
	typename SrtsDetails::T ScanOp(const typename SrtsDetails::T&, const typename SrtsDetails::T&),
	typename SecondarySrtsDetails = typename SrtsDetails::SecondarySrtsDetails>
struct CooperativeGridScan;



/**
 * Cooperative tile scan
 */
template <
	typename SrtsDetails,
	int VEC_SIZE,
	bool EXCLUSIVE,
	typename SrtsDetails::T ScanOp(const typename SrtsDetails::T&, const typename SrtsDetails::T&)>
struct CooperativeTileScan : reduction::CooperativeTileReduction<SrtsDetails, VEC_SIZE, ScanOp>
{
	typedef typename SrtsDetails::T T;

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ScanLane
	{
		static __device__ __forceinline__ void Invoke(
			const SrtsDetails &srts_details,
			T data[SrtsDetails::SCAN_LANES][VEC_SIZE])
		{
			// Retrieve partial reduction from SRTS grid
			T exclusive_partial = srts_details.lane_partial[LANE][0];

			// Scan the partials in this lane/load
			SerialScan<T, VEC_SIZE, EXCLUSIVE, ScanOp>::Invoke(data[LANE], exclusive_partial);

			// Next load
			ScanLane<LANE + 1, TOTAL_LANES>::Invoke(srts_details, data);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ScanLane<TOTAL_LANES, TOTAL_LANES>
	{
		static __device__ __forceinline__ void Invoke(
			const SrtsDetails &srts_details,
			T data[SrtsDetails::SCAN_LANES][VEC_SIZE]) {}
	};


	/**
	 * Scan a single tile where carry is updated with the total aggregate only
	 * in raking threads (homogeneously).
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	static __device__ __forceinline__ void ScanTileWithCarry(
		const SrtsDetails &srts_details,
		T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		CooperativeTileScan::template ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data);

		__syncthreads();

		CooperativeGridScan<SrtsDetails, ScanOp>::ScanTileWithCarry(
			srts_details, carry);

		__syncthreads();

		// Scan partials in tile, retrieving resulting partial from SRTS grid lane partial
		ScanLane<0, SrtsDetails::SCAN_LANES>::Invoke(srts_details, data);
	}


	/**
	 * Scan a single tile.  Total aggregate is computed and returned in all threads.
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	static __device__ __forceinline__ T ScanTile(
		const SrtsDetails &srts_details,
		T data[SrtsDetails::SCAN_LANES][VEC_SIZE])
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		CooperativeTileScan::template ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(
			srts_details, data);

		__syncthreads();

		CooperativeGridScan<SrtsDetails, ScanOp>::ScanTile(
			srts_details);

		__syncthreads();

		// Scan partials in tile, retrieving resulting partial from SRTS grid lane partial
		ScanLane<0, SrtsDetails::SCAN_LANES>::Invoke(srts_details, data);

		// Return last thread's inclusive partial
		return srts_details.CumulativePartial();
	}
};




/******************************************************************************
 * CooperativeGridScan
 ******************************************************************************/

/**
 * Cooperative SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <
	typename SrtsDetails,
	typename SrtsDetails::T ScanOp(const typename SrtsDetails::T&, const typename SrtsDetails::T&)>
struct CooperativeGridScan<SrtsDetails, ScanOp, NullType>
{
	typedef typename SrtsDetails::T T;

	/**
	 * Scan in last-level SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	static __device__ __forceinline__ void ScanTileWithCarry(
		const SrtsDetails &srts_details,
		T &carry)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<T, SrtsDetails::PARTIALS_PER_SEG, ScanOp>::Invoke(
				srts_details.raking_segment);

			// Exclusive warp scan, get total
			T warpscan_total;
			T exclusive_partial = WarpScan<T, SrtsDetails::LOG_RAKING_THREADS, true, SrtsDetails::LOG_RAKING_THREADS, ScanOp>::Invoke(
					inclusive_partial, warpscan_total, srts_details.warpscan);

			// Seed exclusive partial with carry-in
			exclusive_partial = ScanOp(exclusive_partial, carry);

			// Exclusive raking scan
			SerialScan<T, SrtsDetails::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				srts_details.raking_segment, exclusive_partial);

			// Update carry
			carry = ScanOp(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
		}
	}


	/**
	 * Scan in last-level SRTS grid.
	 */
	static __device__ __forceinline__ void ScanTile(const SrtsDetails &srts_details)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T inclusive_partial = reduction::SerialReduce<T, SrtsDetails::PARTIALS_PER_SEG, ScanOp>::Invoke(
				srts_details.raking_segment);

			// Exclusive warp scan
			T exclusive_partial = WarpScan<T, SrtsDetails::LOG_RAKING_THREADS, true, SrtsDetails::LOG_RAKING_THREADS, ScanOp>::Invoke(
				inclusive_partial, srts_details.warpscan);

			// Exclusive raking scan
			SerialScan<T, SrtsDetails::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				srts_details.raking_segment, exclusive_partial);

		}
	}
};


/**
 * Cooperative SRTS grid reduction for multi-level SRTS grids
 */
template <
	typename SrtsDetails,
	typename SrtsDetails::T ScanOp(const typename SrtsDetails::T&, const typename SrtsDetails::T&),
	typename SecondarySrtsDetails>
struct CooperativeGridScan
{
	typedef typename SrtsDetails::T T;

	/**
	 * Scan in SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	static __device__ __forceinline__ void ScanTileWithCarry(
		const SrtsDetails &srts_details,
		T &carry)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<T, SrtsDetails::PARTIALS_PER_SEG, ScanOp>::Invoke(
				srts_details.raking_segment);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondarySrtsDetails, ScanOp>::ScanTileWithCarry(
			srts_details.secondary_details, carry);

		__syncthreads();

		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = srts_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<T, SrtsDetails::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				srts_details.raking_segment, exclusive_partial);
		}
	}


	/**
	 * Scan in SRTS grid.
	 */
	static __device__ __forceinline__ void ScanTile(const SrtsDetails &srts_details)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = reduction::SerialReduce<T, SrtsDetails::PARTIALS_PER_SEG, ScanOp>::Invoke(
				srts_details.raking_segment);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeGridScan<SecondarySrtsDetails, ScanOp>::ScanTile(
			srts_details.secondary_details);

		__syncthreads();

		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			T exclusive_partial = srts_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialScan<T, SrtsDetails::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				srts_details.raking_segment, exclusive_partial);
		}
	}
};



} // namespace scan
} // namespace util
} // namespace b40c

