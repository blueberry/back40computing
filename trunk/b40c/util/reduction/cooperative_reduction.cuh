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
 * Cooperative tile reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/warp_reduce.cuh>

namespace b40c {
namespace util {
namespace reduction {


/**
 * Cooperative reduction in SRTS grid hierarchies
 */
template <
	typename SrtsDetails,
	typename SrtsDetails::T ReductionOp(
		const typename SrtsDetails::T&,
		const typename SrtsDetails::T&),
	typename SecondarySrtsDetails = typename SrtsDetails::SecondarySrtsDetails>
struct CooperativeGridReduction;


/**
 * Cooperative tile reduction
 */
template <
	typename SrtsDetails,
	int VEC_SIZE,
	typename SrtsDetails::T ReductionOp(
		const typename SrtsDetails::T&,
		const typename SrtsDetails::T&)>
struct CooperativeTileReduction
{
	typedef typename SrtsDetails::T T;

	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct ReduceLane
	{
		static __device__ __forceinline__ void Invoke(
			SrtsDetails srts_details,
			T data[SrtsDetails::SCAN_LANES][VEC_SIZE])
		{
			// Reduce the partials in this lane/load
			T partial_reduction = SerialReduce<VEC_SIZE>::template Invoke<T, ReductionOp>(data[LANE]);

			// Store partial reduction into SRTS grid
			srts_details.lane_partial[LANE][0] = partial_reduction;

			// Next load
			ReduceLane<LANE + 1, TOTAL_LANES>::Invoke(srts_details, data);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct ReduceLane<TOTAL_LANES, TOTAL_LANES>
	{
		static __device__ __forceinline__ void Invoke(
			SrtsDetails srts_details,
			T data[SrtsDetails::SCAN_LANES][VEC_SIZE]) {}
	};


	/**
	 * Reduce a single tile.  Carry is computed (or updated if REDUCE_CARRY is set)
	 * only in last raking thread
	 *
	 * Caution: Post-synchronization is needed before srts_details reuse.
	 */
	template <bool REDUCE_CARRY>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsDetails srts_details,
		T data[SrtsDetails::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(srts_details, data);

		__syncthreads();

		CooperativeGridReduction<
			SrtsDetails,
			ReductionOp>::template ReduceTileWithCarry<REDUCE_CARRY>(
				srts_details, carry);
	}

	/**
	 * Reduce a single tile.  Result is computed and returned in all threads.
	 *
	 * No post-synchronization needed before srts_details reuse.
	 */
	static __device__ __forceinline__ T ReduceTile(
		SrtsDetails srts_details,
		T data[SrtsDetails::SCAN_LANES][VEC_SIZE])
	{
		// Reduce partials in tile, placing resulting partial in SRTS grid lane partial
		ReduceLane<0, SrtsDetails::SCAN_LANES>::Invoke(srts_details, data);

		__syncthreads();

		return CooperativeGridReduction<SrtsDetails, ReductionOp>::ReduceTile(
			srts_details);
	}
};




/******************************************************************************
 * CooperativeGridReduction
 ******************************************************************************/

/**
 * Cooperative SRTS grid reduction (specialized for last-level of SRTS grid)
 */
template <
	typename SrtsDetails,
	typename SrtsDetails::T ReductionOp(const typename SrtsDetails::T&, const typename SrtsDetails::T&)>
struct CooperativeGridReduction<SrtsDetails, ReductionOp, NullType>
{
	typedef typename SrtsDetails::T T;

	/**
	 * Reduction in last-level SRTS grid.  Carry is computed (or updated if REDUCE_CARRY is set)
	 * only in last raking thread
	 */
	template <bool REDUCE_CARRY>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsDetails srts_details,
		T &carry)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::template Invoke<T, ReductionOp>(
				srts_details.raking_segment);

			// Warp reduction
			T warpscan_total = WarpReduce<SrtsDetails::LOG_RAKING_THREADS>::template InvokeSingle<T, ReductionOp>(
				partial, srts_details.warpscan);

			carry = (REDUCE_CARRY) ?
				ReductionOp(carry, warpscan_total) : 	// Update carry
				warpscan_total;
		}
	}


	/**
	 * Reduction in last-level SRTS grid.  Result is computed in all threads.
	 */
	static __device__ __forceinline__ T ReduceTile(SrtsDetails srts_details)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::template Invoke<T, ReductionOp>(
				srts_details.raking_segment);

			// Warp reduction
			WarpReduce<SrtsDetails::LOG_RAKING_THREADS>::template InvokeSingle<T, ReductionOp>(
				partial, srts_details.warpscan);
		}

		__syncthreads();

		// Return last thread's inclusive partial
		return srts_details.CumulativePartial();
	}
};


/**
 * Cooperative SRTS grid reduction for multi-level SRTS grids
 */
template <
	typename SrtsDetails,
	typename SrtsDetails::T ReductionOp(const typename SrtsDetails::T&, const typename SrtsDetails::T&),
	typename SecondarySrtsDetails>
struct CooperativeGridReduction
{
	typedef typename SrtsDetails::T T;

	/**
	 * Reduction in SRTS grid.  Carry-in/out is updated only in raking threads (homogeneously)
	 */
	template <bool REDUCE_CARRY>
	static __device__ __forceinline__ void ReduceTileWithCarry(
		SrtsDetails srts_details,
		T &carry)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::template Invoke<T, ReductionOp>(
				srts_details.raking_segment);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		CooperativeGridReduction<
			SecondarySrtsDetails,
			ReductionOp>::template ReduceTileWithCarry<REDUCE_CARRY>(
					srts_details.secondary_details, carry);
	}


	/**
	 * Reduction in SRTS grid.  Result is computed in all threads.
	 */
	static __device__ __forceinline__ T ReduceTile(SrtsDetails srts_details)
	{
		if (threadIdx.x < SrtsDetails::RAKING_THREADS) {

			// Raking reduction
			T partial = SerialReduce<SrtsDetails::PARTIALS_PER_SEG>::template Invoke<T, ReductionOp>(
				srts_details.raking_segment);

			// Place partial in next grid
			srts_details.secondary_details.lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Collectively reduce in next grid
		return CooperativeGridReduction<SecondarySrtsDetails, ReductionOp>::ReduceTile(
			srts_details.secondary_details);
	}
};



} // namespace reduction
} // namespace util
} // namespace b40c

