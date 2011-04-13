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
 * Cooperative SOA tile reduction and scanning within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/soa/cooperative_soa_reduction.cuh>
#include <b40c/util/scan/soa/serial_soa_scan.cuh>
#include <b40c/util/scan/soa/warp_soa_scan.cuh>

namespace b40c {
namespace util {
namespace scan {
namespace soa {

/**
 * Cooperative SOA reduction in SRTS grid hierarchies
 */
template <
	typename SrtsSoaDetails,
	typename SrtsSoaDetails::SoaTuple ScanOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&),
	typename SecondarySrtsSoaDetails = typename SrtsSoaDetails::SecondarySrtsSoaDetails>
struct CooperativeSoaGridScan;



/**
 * Cooperative SOA tile scan
 */
template <
	typename SrtsSoaDetails,
	int VEC_SIZE,
	bool EXCLUSIVE,
	typename SrtsSoaDetails::SoaTuple ScanOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&),
	typename SrtsSoaDetails::SoaTuple FinalSoaScanOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&)>
struct CooperativeSoaTileScan :
	reduction::soa::CooperativeSoaTileReduction<SrtsSoaDetails, VEC_SIZE, ScanOp>		// Inherit from cooperative tile reduction
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	// Next lane/load
	template <int LANE, int TOTAL_LANES, typename DataSoa>
	struct ScanLane
	{
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			DataSoa data_soa)
		{
			// Retrieve partial reduction from SRTS grid
			SoaTuple exclusive_partial = srts_soa_details.lane_partials.template Get<LANE, SoaTuple>(0);

			// Scan the partials in this lane/load using the FinalSoaScanOp
			SerialSoaScanLane<
				SoaTuple,
				DataSoa,
				LANE,
				VEC_SIZE,
				EXCLUSIVE,
				FinalSoaScanOp>::Invoke(data_soa, exclusive_partial);

			// Next load
			ScanLane<LANE + 1, TOTAL_LANES, DataSoa>::Invoke(srts_soa_details, data_soa);
		}
	};

	// Terminate
	template <int TOTAL_LANES, typename DataSoa>
	struct ScanLane<TOTAL_LANES, TOTAL_LANES, DataSoa>
	{
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			DataSoa data_soa) {}
	};


	/**
	 * Scan a single tile where carry is updated with the total aggregate only
	 * in raking threads (homogeneously).
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	template <typename DataSoa>
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		DataSoa data_soa,
		SoaTuple &carry)
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		CooperativeSoaTileScan::template ReduceLane<0, SrtsSoaDetails::SCAN_LANES, DataSoa>::Invoke(
			srts_soa_details, data_soa);

		__syncthreads();

		CooperativeSoaGridScan<SrtsSoaDetails, ScanOp>::ScanTileWithCarry(
			srts_soa_details, carry);

		__syncthreads();

		// Scan partials in tile, retrieving resulting partial from SRTS grid lane partial
		ScanLane<0, SrtsSoaDetails::SCAN_LANES, DataSoa>::Invoke(
			srts_soa_details, data_soa);
	}


	/**
	 * Scan a single tile.  Total aggregate is computed and returned in all threads.
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	template <typename DataSoa>
	static __device__ __forceinline__ SoaTuple ScanTile(
		SrtsSoaDetails srts_soa_details,
		DataSoa data_soa)
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		CooperativeSoaTileScan::template ReduceLane<0, SrtsSoaDetails::SCAN_LANES, DataSoa>::Invoke(
			srts_soa_details, data_soa);

		__syncthreads();

		CooperativeSoaGridScan<SrtsSoaDetails, ScanOp>::ScanTile(
			srts_soa_details);

		__syncthreads();

		// Scan partials in tile, retrieving resulting partial from SRTS grid lane partial
		ScanLane<0, SrtsSoaDetails::SCAN_LANES, DataSoa>::Invoke(
			srts_soa_details, data_soa);

		// Return last thread's inclusive partial
		return srts_soa_details.CumulativePartial();
	}
};




/******************************************************************************
 * CooperativeSoaGridScan
 ******************************************************************************/

/**
 * Cooperative SOA SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <
	typename SrtsSoaDetails,
	typename SrtsSoaDetails::SoaTuple ScanOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&)>
struct CooperativeSoaGridScan<SrtsSoaDetails, ScanOp, NullType>
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	/**
	 * Scan in last-level SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		SoaTuple &carry)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple inclusive_partial = reduction::soa::SerialSoaReduce<
				SoaTuple,
				typename SrtsSoaDetails::RakingSoa,
				SrtsSoaDetails::PARTIALS_PER_SEG,
				ScanOp>::Invoke(srts_soa_details.raking_segments);

			// Exclusive warp scan, get total
			SoaTuple warpscan_total;
			SoaTuple exclusive_partial = WarpSoaScan<
				SoaTuple,
				typename SrtsSoaDetails::WarpscanSoa,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				true,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				ScanOp>::Invoke(inclusive_partial, warpscan_total, srts_soa_details.warpscan_partials);

			// Seed exclusive partial with carry-in
			exclusive_partial = ScanOp(carry, exclusive_partial);

			// Exclusive raking scan
			SerialSoaScan<
				SoaTuple,
				typename SrtsSoaDetails::RakingSoa,
				SrtsSoaDetails::PARTIALS_PER_SEG,
				true,
				ScanOp>::Invoke(srts_soa_details.raking_segments, exclusive_partial);

			// Update carry
			carry = ScanOp(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
		}
	}


	/**
	 * Scan in last-level SRTS grid.
	 */
	static __device__ __forceinline__ void ScanTile(
		SrtsSoaDetails srts_soa_details)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple inclusive_partial = reduction::soa::SerialSoaReduce<
				SoaTuple,
				typename SrtsSoaDetails::RakingSoa,
				SrtsSoaDetails::PARTIALS_PER_SEG,
				ScanOp>::Invoke(srts_soa_details.raking_segments);

			// Exclusive warp scan
			SoaTuple exclusive_partial = WarpSoaScan<
				SoaTuple,
				typename SrtsSoaDetails::WarpscanSoa,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				true,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				ScanOp>::Invoke(inclusive_partial, srts_soa_details.warpscan_partials);

			// Exclusive raking scan
			SerialSoaScan<
				SoaTuple,
				typename SrtsSoaDetails::RakingSoa,
				SrtsSoaDetails::PARTIALS_PER_SEG,
				true,
				ScanOp>::Invoke(srts_soa_details.raking_segments, exclusive_partial);
		}
	}
};


/**
 * Cooperative SOA SRTS grid reduction for multi-level SRTS grids
 * /
template <
	typename SrtsSoaDetails,
	typename SrtsSoaDetails::SoaTuple ScanOp(typename SrtsSoaDetails::SoaTuple&, typename SrtsSoaDetails::SoaTuple&),
	typename SecondarySrtsSoaDetails>
struct CooperativeSoaGridScan
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	/ **
	 * Scan in SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 * /
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsSoaDetails &srts_soa_details,
		SoaTuple &carry)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple partial = reduction::SerialReduce<SoaTuple, SrtsSoaDetails::PARTIALS_PER_SEG, ScanOp>::Invoke(
				srts_soa_details.raking_segment);

			// Place partial in next grid
			srts_soa_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeSoaGridScan<SecondarySrtsSoaDetails, ScanOp>::ScanTileWithCarry(
			srts_soa_details.secondary_details, carry);

		__syncthreads();

		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			SoaTuple exclusive_partial = srts_soa_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialSoaScan<SoaTuple, SrtsSoaDetails::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				srts_soa_details.raking_segment, exclusive_partial);
		}
	}


	/ **
	 * Scan in SRTS grid.
	 * /
	static __device__ __forceinline__ void ScanTile(SrtsSoaDetails &srts_soa_details)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple partial = reduction::SerialReduce<SoaTuple, SrtsSoaDetails::PARTIALS_PER_SEG, ScanOp>::Invoke(
				srts_soa_details.raking_segment);

			// Place partial in next grid
			srts_soa_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively scan in next grid
		CooperativeSoaGridScan<SecondarySrtsSoaDetails, ScanOp>::ScanTile(
			srts_soa_details.secondary_details);

		__syncthreads();

		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Retrieve partial from next grid
			SoaTuple exclusive_partial = srts_soa_details.secondary_details.lane_partial[0][0];

			// Exclusive raking scan
			SerialSoaScan<SoaTuple, SrtsSoaDetails::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				srts_soa_details.raking_segment, exclusive_partial);
		}
	}
};
*/

} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

