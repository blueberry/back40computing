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
 * Base class for tile reduction within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/reduction/reduction_utils.cuh>

namespace b40c {
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
	typename SrtsGrid,
	int VEC_SIZE,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
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

	// External storage to use for warpscan
	T (*warpscan)[B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	// Pointers for local raking in a single SRTS grid
	T *primary_base_partial;
	T *primary_raking_seg;


	/**
	 * Reduce a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <int VEC_SIZE, T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ void ReduceTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(data, primary_base_partial);

		__syncthreads();

		// Rake and reduce in primary SRTS grid
		WarpRakeAndReduce<SrtsGrid, BinaryOp>(primary_raking_seg, warpscan, carry);

		__syncthreads();
	}


	/**
	 * Reduce a single tile.  Result is computed in all threads.
	 */
	template <int VEC_SIZE, T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ T ReduceTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(data, primary_base_partial);

		__syncthreads();

		// Rake and reduce in primary SRTS grid
		return WarpRakeAndReduce<SrtsGrid, BinaryOp>(primary_raking_seg, warpscan);
	}

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CollectiveReduction(
		uint4 smem_pool[SrtsGrid::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)]) :
			warpscan(warpscan),
			primary_base_partial(SrtsGrid::BasePartial(reinterpret_cast<T*>(smem_pool))),
			primary_raking_seg(NULL)
	{
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			primary_raking_seg = SrtsGrid::RakingSegment(reinterpret_cast<T*>(smem_pool));
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

	// External storage to use for warpscan
	T (*warpscan)[B40C_WARP_THREADS(__B40C_CUDA_ARCH__)];

	// Pointers for local raking in a primary SRTS grid
	T *primary_base_partial;
	T *primary_raking_seg;

	// Pointers for local raking in a secondary SRTS grid
	T *secondary_base_partial;
	T *secondary_raking_seg;


	/**
	 * Reduce a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <int VEC_SIZE, T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ void ReduceTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(data, primary_base_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(primary_raking_seg);
			secondary_base_partial[0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan
		WarpRakeAndReduce<SecondarySrtsGrid, BinaryOp>(secondary_raking_seg, warpscan, carry);

		__syncthreads();
	}


	/**
	 * Reduce a single tile.  Result is computed in all threads.
	 */
	template <int VEC_SIZE, T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ T ReduceTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(data, primary_base_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(primary_raking_seg);
			secondary_base_partial[0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan
		return WarpRakeAndReduce<SecondarySrtsGrid, BinaryOp>(secondary_raking_seg, warpscan);
	}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ CollectiveReduction(
		uint4 smem_pool[SrtsGrid::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)]) :
			warpscan(warpscan),
			primary_base_partial(SrtsGrid::BasePartial(reinterpret_cast<T*>(smem_pool))),
			primary_raking_seg(NULL),
			secondary_base_partial(NULL),
			secondary_raking_seg(NULL)
	{
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

			T *secondary_grid = reinterpret_cast<T*>(smem_pool + SrtsGrid::SMEM_QUADS);

			primary_raking_seg = SrtsGrid::RakingSegment(reinterpret_cast<T*>(smem_pool));
			secondary_base_partial = SecondarySrtsGrid::BasePartial(secondary_grid);

			if (threadIdx.x < SecondarySrtsGrid::RAKING_THREADS) {
				secondary_raking_seg = SecondarySrtsGrid::RakingSegment(secondary_grid);
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
	typename SrtsGrid,
	int VEC_SIZE,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
struct ReduceVectors
{
	typedef typename SrtsGrid::T T;

	// Next load
	template <int LOAD, int TOTAL_LOADS>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
			T *base_partial)
		{
			// Store partial reduction into SRTS grid
			T partial_reduction = reduction::SerialReduce<T, VEC_SIZE, BinaryOp>::Invoke(data[LOAD]);
			base_partial[LOAD * SrtsGrid::LANE_STRIDE] = partial_reduction;

			// Next load
			Iterate<LOAD + 1, TOTAL_LOADS>::Invoke(data, base_partial);
		}
	};

	// Terminate
	template <int TOTAL_LOADS>
	struct Iterate<TOTAL_LOADS, TOTAL_LOADS> {
		static __device__ __forceinline__ void Invoke(
			T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
			T *base_partial) {}
	};

	// Interface
	static __device__ __forceinline__ void Invoke(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T *base_partial)
	{
		Iterate<0, SrtsGrid::SCAN_LANES>::Invoke(data, base_partial);
	}
};




} // namespace reduction
} // namespace b40c

