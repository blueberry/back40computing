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
 * Base class for tile reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/warp_rake_and_reduce.cuh>

namespace b40c {
namespace util {
namespace reduction {


/******************************************************************************
 * CollectiveReduction Declaration
 ******************************************************************************/

/**
 * Base class for tile-reduction routines
 */
template <
	typename SrtsGrid,
	typename SecondarySrtsGrid = typename SrtsGrid::SecondaryGrid>
struct CollectiveReduction;


/**
 * Helper structure for reducing each load in registers and placing into smem
 */
template <
	typename T,
	T ReductionOp(const T&, const T&),
	typename LanePartial,
	int SCAN_LANES,
	int VEC_SIZE>
struct ReduceVectors;


/******************************************************************************
 * CollectiveReduction Implementation  (specialized for one-level SRTS grid)
 ******************************************************************************/

/**
 * Base class for tile-reduction routines (specialized for one-level SRTS grid)
 */
template <typename SrtsGrid>
struct CollectiveReduction<SrtsGrid, util::InvalidSrtsGrid>
{
	typedef typename SrtsGrid::T T;
	typedef typename SrtsGrid::LanePartial PrimaryLanePartial;

	// External storage to use for warpscan
	T (*warpscan)[B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	// Pointers for local raking in a single SRTS grid
	PrimaryLanePartial primary_lane_partial;
	T *primary_raking_seg;


	/**
	 * Reduce a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <int VEC_SIZE, T ReductionOp(const T&, const T&)>
	__device__ __forceinline__ void ReduceTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<T, ReductionOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, primary_lane_partial);

		__syncthreads();

		// Rake and reduce in primary SRTS grid
		WarpRakeAndReduce<T, ReductionOp, SrtsGrid::PARTIALS_PER_SEG, SrtsGrid::LOG_RAKING_THREADS>(
			primary_raking_seg, warpscan, carry);

		__syncthreads();
	}


	/**
	 * Reduce a single tile.  Result is computed in all threads.
	 */
	template <int VEC_SIZE, T ReductionOp(const T&, const T&)>
	__device__ __forceinline__ T ReduceTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<T, ReductionOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, primary_lane_partial);

		__syncthreads();

		// Rake and reduce in primary SRTS grid
		return WarpRakeAndReduce<T, ReductionOp, SrtsGrid::PARTIALS_PER_SEG, SrtsGrid::LOG_RAKING_THREADS>(
			primary_raking_seg, warpscan);
	}

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CollectiveReduction(
		uint4 smem_pool[SrtsGrid::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)]) :
			warpscan(warpscan),
			primary_lane_partial(SrtsGrid::MyLanePartial(reinterpret_cast<T*>(smem_pool))),
			primary_raking_seg(NULL)
	{
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			primary_raking_seg = SrtsGrid::MyRakingSegment(reinterpret_cast<T*>(smem_pool));
		}
	}
};


/******************************************************************************
 * CollectiveReduction Implementation (specialized for two-level SRTS grid)
 ******************************************************************************/

/**
 * Base class for tile-reduction routines (specialized for two-level SRTS grid)
 *
 * Extends CollectiveReduction for one-level SRTS grid
 */
template <typename SrtsGrid, typename SecondarySrtsGrid>
struct CollectiveReduction
{
	typedef typename SrtsGrid::T T;
	typedef typename SrtsGrid::LanePartial PrimaryLanePartial;
	typedef typename SecondarySrtsGrid::LanePartial SecondaryLanePartial;

	// External storage to use for warpscan
	T (*warpscan)[B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	// Pointers for local raking in a primary SRTS grid
	PrimaryLanePartial primary_lane_partial;
	T *primary_raking_seg;

	// Pointers for local raking in a secondary SRTS grid
	SecondaryLanePartial secondary_lane_partial;
	T *secondary_raking_seg;


	/**
	 * Reduce a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <int VEC_SIZE, T ReductionOp(const T&, const T&)>
	__device__ __forceinline__ void ReduceTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<T, ReductionOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, primary_lane_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, ReductionOp>::Invoke(primary_raking_seg);
			secondary_lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan
		WarpRakeAndReduce<T, ReductionOp, SrtsGrid::PARTIALS_PER_SEG, SecondarySrtsGrid::LOG_RAKING_THREADS>(
			secondary_raking_seg, warpscan, carry);

		__syncthreads();
	}


	/**
	 * Reduce a single tile.  Result is computed in all threads.
	 */
	template <int VEC_SIZE, T ReductionOp(const T&, const T&)>
	__device__ __forceinline__ T ReduceTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<T, ReductionOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
				data, primary_lane_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, ReductionOp>::Invoke(primary_raking_seg);
			secondary_lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan
		return WarpRakeAndReduce<T, ReductionOp, SrtsGrid::PARTIALS_PER_SEG, SecondarySrtsGrid::LOG_RAKING_THREADS>(
			secondary_raking_seg, warpscan);
	}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ CollectiveReduction(
		uint4 smem_pool[SrtsGrid::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)]) :
			warpscan(warpscan),
			primary_lane_partial(SrtsGrid::MyLanePartial(reinterpret_cast<T*>(smem_pool))),
			primary_raking_seg(NULL),
			secondary_lane_partial(NULL),
			secondary_raking_seg(NULL)
	{
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

			T *secondary_grid = reinterpret_cast<T*>(smem_pool + SrtsGrid::PRIMARY_SMEM_QUADS);

			primary_raking_seg = SrtsGrid::MyRakingSegment(reinterpret_cast<T*>(smem_pool));
			secondary_lane_partial = SecondarySrtsGrid::MyLanePartial(secondary_grid);

			if (threadIdx.x < SecondarySrtsGrid::RAKING_THREADS) {
				secondary_raking_seg = SecondarySrtsGrid::MyRakingSegment(secondary_grid);
			}
		}
	}
};


/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Reduce each load in registers and place into smem
 */
template <
	typename T,
	T ReductionOp(const T&, const T&),
	typename LanePartial,
	int SCAN_LANES,
	int VEC_SIZE>
struct ReduceVectors
{
	// Next lane/load
	template <int LANE, int TOTAL_LANES>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			T data[SCAN_LANES][VEC_SIZE],
			LanePartial lane_partial)
		{
			// Reduce the partials in this lane/load
			T partial_reduction = SerialReduce<T, VEC_SIZE, ReductionOp>::Invoke(data[LANE]);

			// Store partial reduction into SRTS grid
			lane_partial[LANE][0] = partial_reduction;

			// Next load
			Iterate<LANE + 1, TOTAL_LANES>::Invoke(data, lane_partial);
		}
	};

	// Terminate
	template <int TOTAL_LANES>
	struct Iterate<TOTAL_LANES, TOTAL_LANES> {
		static __device__ __forceinline__ void Invoke(
			T data[SCAN_LANES][VEC_SIZE],
			LanePartial lane_partial) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		T data[SCAN_LANES][VEC_SIZE],
		LanePartial lane_partial)
	{
		Iterate<0, SCAN_LANES>::Invoke(data, lane_partial);
	}
};




} // namespace reduction
} // namespace util
} // namespace b40c

