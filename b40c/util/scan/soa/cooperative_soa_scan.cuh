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
 * Cooperative tile SOA (structure-of-arrays) scan within CTAs
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
 * Cooperative SOA reduction in SRTS smem grid hierarchies
 */
template <
	typename SrtsSoaDetails,
	typename SecondarySrtsSoaDetails = typename SrtsSoaDetails::SecondarySrtsSoaDetails>
struct CooperativeSoaGridScan;



/**
 * Cooperative SOA tile scan
 */
template <
	int VEC_SIZE,
	bool EXCLUSIVE>					// Whether or not this is an exclusive scan
struct CooperativeSoaTileScan
{
	//---------------------------------------------------------------------
	// Iteration structures for extracting partials from SRTS lanes and
	// using them to seed scans of tile vectors
	//---------------------------------------------------------------------

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ScanLane
	{
		template <
			typename SrtsSoaDetails,
			typename DataSoa,
			typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			DataSoa data_soa,
			ReductionOp scan_op)
		{
			// Retrieve partial reduction from SRTS grid
			typename SrtsSoaDetails::SoaTuple exclusive_partial;
			srts_soa_details.lane_partials.Get(exclusive_partial, LANE, 0);

			// Scan the partials in this lane/load
			SerialSoaScan<VEC_SIZE, EXCLUSIVE>::Scan(
				data_soa, exclusive_partial, LANE, scan_op);

			// Next load
			ScanLane<LANE + 1, TOTAL_LANES>::Invoke(
				srts_soa_details, data_soa, scan_op);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ScanLane<TOTAL_LANES, TOTAL_LANES>
	{
		template <
			typename SrtsSoaDetails,
			typename DataSoa,
			typename ReductionOp>
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			DataSoa data_soa,
			ReductionOp scan_op) {}
	};


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------

	/**
	 * Scan a single tile where carry is updated with the total aggregate only
	 * in raking threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <
		typename SrtsSoaDetails,
		typename DataSoa,
		typename SoaTuple,
		typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		DataSoa data_soa,
		SoaTuple &carry,
		ReductionOp scan_op)
	{
		// Reduce vectors in tile, placing resulting partial into corresponding SRTS grid lanes
		reduction::soa::CooperativeSoaTileReduction<VEC_SIZE>::template
			ReduceLane<0, SrtsSoaDetails::SCAN_LANES>::Invoke(
				srts_soa_details, data_soa, scan_op);

		__syncthreads();

		CooperativeSoaGridScan<SrtsSoaDetails>::ScanTileWithCarry(
			srts_soa_details, carry, scan_op);

		__syncthreads();

		// Scan partials in tile, retrieving resulting partial from SRTS grid lane partial
		ScanLane<0, SrtsSoaDetails::SCAN_LANES>::Invoke(
			srts_soa_details, data_soa, scan_op);
	}


	/**
	 * Scan a single tile.  Total aggregate is computed and returned in all threads.
	 *
	 * No post-synchronization needed before grid reuse.
	 */
	template <
		typename SrtsSoaDetails,
		typename DataSoa,
		typename SoaTuple,
		typename ReductionOp>
	static __device__ __forceinline__ void ScanTile(
		SoaTuple &retval,
		SrtsSoaDetails srts_soa_details,
		DataSoa data_soa,
		ReductionOp scan_op)
	{
		// Reduce vectors in tile, placing resulting partial into corresponding SRTS grid lanes
		reduction::soa::CooperativeSoaTileReduction<VEC_SIZE>::template
			ReduceLane<0, SrtsSoaDetails::SCAN_LANES>::Invoke(
				srts_soa_details, data_soa, scan_op);

		__syncthreads();

		CooperativeSoaGridScan<SrtsSoaDetails>::ScanTile(
			srts_soa_details, scan_op);

		__syncthreads();

		// Scan partials in tile, retrieving resulting partial from SRTS grid lane partial
		ScanLane<0, SrtsSoaDetails::SCAN_LANES>::Invoke(
			srts_soa_details, data_soa, scan_op);

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
template <typename SrtsSoaDetails>
struct CooperativeSoaGridScan<SrtsSoaDetails, NullType>
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	/**
	 * Scan in last-level SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		SoaTuple &carry,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple inclusive_partial;
			reduction::soa::SerialSoaReduce<SrtsSoaDetails::PARTIALS_PER_SEG>::Reduce(
				inclusive_partial,
				srts_soa_details.raking_segments,
				scan_op);

			// Exclusive warp scan, get total
			SoaTuple warpscan_total;
			SoaTuple exclusive_partial = WarpSoaScan<SrtsSoaDetails::LOG_RAKING_THREADS>::Scan(
				inclusive_partial,
				warpscan_total,
				srts_soa_details.warpscan_partials,
				scan_op);

			// Seed exclusive partial with carry-in
			exclusive_partial = scan_op(carry, exclusive_partial);

			// Exclusive raking scan
			SerialSoaScan<SrtsSoaDetails::PARTIALS_PER_SEG>::Scan(
				srts_soa_details.raking_segments, exclusive_partial, scan_op);

			// Update carry
			carry = scan_op(carry, warpscan_total);			// Increment the CTA's running total by the full tile reduction
		}
	}


	/**
	 * Scan in last-level SRTS grid.
	 */
	template <typename ReductionOp>
	static __device__ __forceinline__ void ScanTile(
		SrtsSoaDetails srts_soa_details,
		ReductionOp scan_op)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple inclusive_partial;
			reduction::soa::SerialSoaReduce<SrtsSoaDetails::PARTIALS_PER_SEG>::Reduce(
				inclusive_partial,
				srts_soa_details.raking_segments,
				scan_op);

			// Exclusive warp scan
			SoaTuple exclusive_partial = WarpSoaScan<SrtsSoaDetails::LOG_RAKING_THREADS>::Scan(
				inclusive_partial,
				srts_soa_details.warpscan_partials,
				scan_op);

			// Exclusive raking scan
			SerialSoaScan<SrtsSoaDetails::PARTIALS_PER_SEG>::Scan(
				srts_soa_details.raking_segments, exclusive_partial, scan_op);
		}
	}
};


} // namespace soa
} // namespace scan
} // namespace util
} // namespace b40c

