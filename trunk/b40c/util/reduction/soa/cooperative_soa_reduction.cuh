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
 * Cooperative SOA tile SOA (structure-of-arrays) reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/soa/serial_soa_reduce.cuh>
#include <b40c/util/scan/soa/warp_soa_scan.cuh>

namespace b40c {
namespace util {
namespace reduction {
namespace soa {


/**
 * Cooperative SOA reduction in SRTS grid hierarchies
 */
template <
	typename SrtsSoaDetails,
	typename SrtsSoaDetails::SoaTuple ReductionOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&),
	typename SecondarySrtsSoaDetails = typename SrtsSoaDetails::SecondarySrtsSoaDetails>
struct CooperativeSoaGridReduction;


/**
 * Cooperative SOA tile reduction
 */
template <
	typename SrtsSoaDetails,
	int VEC_SIZE,
	typename SrtsSoaDetails::SoaTuple ReductionOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&)>
struct CooperativeSoaTileReduction
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	// Next lane/load
	template <int LANE, int TOTAL_LANES, typename DataSoa>
	struct ReduceLane
	{
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			DataSoa data_soa)
		{
			// Reduce the partials in this lane/load
			SoaTuple partial_reduction = SerialSoaReduceLane<
					SoaTuple, DataSoa, LANE, VEC_SIZE, ReductionOp>::Invoke(data_soa);

			// Store partial reduction into SRTS grid
			srts_soa_details.lane_partials.template Set<LANE>(partial_reduction, 0);

			// Next load
			ReduceLane<LANE + 1, TOTAL_LANES, DataSoa>::Invoke(srts_soa_details, data_soa);
		}
	};

	// Terminate
	template <int TOTAL_LANES, typename DataSoa>
	struct ReduceLane<TOTAL_LANES, TOTAL_LANES, DataSoa>
	{
		static __device__ __forceinline__ void Invoke(
			SrtsSoaDetails srts_soa_details,
			DataSoa data_soa) {}
	};


	/**
	 * Reduce a single tile.  Carry is computed (or updated if REDUCE_CARRY is set)
	 * only in last raking thread
	 *
	 * Caution: Post-synchronization is needed before srts_details reuse.
	 */
	template <typename DataSoa>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		DataSoa data_soa,
		SoaTuple &carry)
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		ReduceLane<0, SrtsSoaDetails::SCAN_LANES, DataSoa>::Invoke(
			srts_soa_details, data_soa);

		__syncthreads();

		CooperativeSoaGridReduction<SrtsSoaDetails, ReductionOp>::template ReduceTileWithCarry<true>(
			srts_soa_details, carry);
	}

	/**
	 * Reduce a single tile.  Result is computed and returned in all threads.
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	template <typename DataSoa>
	static __device__ __forceinline__ SoaTuple ReduceTile(
		SrtsSoaDetails srts_soa_details,
		DataSoa data_soa)
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		ReduceLane<0, SrtsSoaDetails::SCAN_LANES, DataSoa>::Invoke(
			srts_soa_details, data_soa);

		__syncthreads();

		return CooperativeSoaGridReduction<SrtsSoaDetails, ReductionOp>::ReduceTile(
			srts_soa_details);
	}
};




/******************************************************************************
 * CooperativeSoaGridReduction
 ******************************************************************************/

/**
 * Cooperative SOA SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <
	typename SrtsSoaDetails,
	typename SrtsSoaDetails::SoaTuple ReductionOp(
		typename SrtsSoaDetails::SoaTuple&,
		typename SrtsSoaDetails::SoaTuple&)>
struct CooperativeSoaGridReduction<SrtsSoaDetails, ReductionOp, NullType>
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	/**
	 * Reduction in last-level SRTS grid.  Carry is computed (or updated if REDUCE_CARRY is set)
	 * only in last raking thread
	 */
	template <bool REDUCE_CARRY>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsSoaDetails srts_soa_details,
		SoaTuple &carry)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple inclusive_partial = SerialSoaReduce<
				SoaTuple,
				typename SrtsSoaDetails::RakingSoa,
				SrtsSoaDetails::PARTIALS_PER_SEG,
				ReductionOp>::Invoke(srts_soa_details.raking_segments);

			// Inclusive warp scan that sets warpscan total in all
			// raking threads
			SoaTuple warpscan_total;
			scan::soa::WarpSoaScan<
				SoaTuple,
				typename SrtsSoaDetails::WarpscanSoa,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				false,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				ReductionOp>::Invoke(
					inclusive_partial, warpscan_total, srts_soa_details.warpscan_partials);

			carry = (REDUCE_CARRY) ?
				ReductionOp(carry, warpscan_total) : 	// Update carry
				warpscan_total;
		}
	}


	/**
	 * Reduction in last-level SRTS grid.  Result is computed in all threads.
	 */
	static __device__ __forceinline__ SoaTuple ReduceTile(
		SrtsSoaDetails srts_soa_details)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple inclusive_partial = SerialSoaReduce<
				SoaTuple,
				typename SrtsSoaDetails::RakingSoa,
				SrtsSoaDetails::PARTIALS_PER_SEG,
				ReductionOp>::Invoke(srts_soa_details.raking_segments);

			// Inclusive warp scan that sets warpscan total in all
			// raking threads
			SoaTuple warpscan_total;
			scan::soa::WarpSoaScan<
				SoaTuple,
				typename SrtsSoaDetails::WarpscanSoa,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				false,
				SrtsSoaDetails::LOG_RAKING_THREADS,
				ReductionOp>::Invoke(
					inclusive_partial, warpscan_total, srts_soa_details.warpscan_partials);
		}

		__syncthreads();

		// Return last thread's inclusive partial
		return srts_soa_details.CumulativePartial();
	}
};


/**
 * Cooperative SOA SRTS grid reduction for multi-level SRTS grids
 * /
template <
	typename SrtsSoaDetails,
	typename SrtsSoaDetails::SoaTuple ReductionOp(typename SrtsSoaDetails::SoaTuple&, typename SrtsSoaDetails::SoaTuple&),
	typename SecondarySrtsSoaDetails>
struct CooperativeSoaGridReduction
{
	typedef typename SrtsSoaDetails::SoaTuple SoaTuple;

	/ **
	 * Reduction in SRTS grid.  Carry is computed (or updated if REDUCE_CARRY is set)
	 * only in last raking thread
	 * /
	template <bool REDUCE_CARRY>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsSoaDetails &srts_soa_details,
		SoaTuple &carry)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple partial = SerialReduce<SoaTuple, SrtsSoaDetails::PARTIALS_PER_SEG, ReductionOp>::Invoke(
				srts_soa_details.raking_segment);

			// Place partial in next grid
			srts_soa_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		CooperativeSoaGridReduction<SecondarySrtsSoaDetails, ReductionOp>::ReduceTileWithCarry(
			srts_soa_details.secondary_details, carry);
	}


	/ **
	 * Reduction in SRTS grid.  Result is computed in all threads.
	 * /
	static __device__ __forceinline__ SoaTuple ReduceTile(SrtsSoaDetails &srts_soa_details)
	{
		if (threadIdx.x < SrtsSoaDetails::RAKING_THREADS) {

			// Raking reduction
			SoaTuple partial = SerialReduce<SoaTuple, SrtsSoaDetails::PARTIALS_PER_SEG, ReductionOp>::Invoke(
				srts_soa_details.raking_segment);

			// Place partial in next grid
			srts_soa_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		return CooperativeSoaGridReduction<SecondarySrtsSoaDetails, ReductionOp>::ReduceTile(
			srts_soa_details.secondary_details);
	}
};
*/

} // namespace soa
} // namespace reduction
} // namespace util
} // namespace b40c

