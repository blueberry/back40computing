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
 * Base class for tile-reduction and scanning within CTAs
 ******************************************************************************/

#pragma once

#include <b40c/util/srts_grid.cuh>
#include <b40c/reduction/collective_reduction.cuh>
#include <b40c/scan/scan_utils.cuh>

namespace b40c {
namespace scan {


/******************************************************************************
 * CollectiveScan Declaration
 ******************************************************************************/

/**
 * Base class for tile-scanning routines
 */
template <
	typename SrtsGrid,
	typename SecondarySrtsGrid = typename SrtsGrid::SecondaryGrid>
struct CollectiveScan;


/**
 * Helper structure for scanning each load in registers, seeding from smem partials
 */
template <
	typename SrtsGrid,
	int VEC_SIZE,
	bool EXCLUSIVE,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
struct ScanVectors;



/******************************************************************************
 * CollectiveScan Implementation (specialized for one-level SRTS grid)
 ******************************************************************************/


/**
 * Base class for tile-scanning routines (specialized for one-level SRTS grid)
 *
 * Extends CollectiveReduction for one-level SRTS grids
 */
template <typename SrtsGrid>
struct CollectiveScan<SrtsGrid, util::InvalidSrtsGrid> : reduction::CollectiveReduction<SrtsGrid>
{
	typedef typename SrtsGrid::T T;

	/**
	 * Scan a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ void ScanTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(
			data, this->primary_base_partial);

		__syncthreads();

		// Primary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<SrtsGrid, true, BinaryOp>(
			this->primary_raking_seg, this->warpscan, carry);

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<SrtsGrid, VEC_SIZE, EXCLUSIVE, BinaryOp>::Invoke(
			data, this->primary_base_partial);
	}


	/**
	 * Scan a single tile.  Total reduction is returned to all threads
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ T ScanTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(
			data, this->primary_base_partial);

		__syncthreads();

		// Primary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<SrtsGrid, true, BinaryOp>(
			this->primary_raking_seg, this->warpscan);

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<SrtsGrid, VEC_SIZE, EXCLUSIVE, BinaryOp>::Invoke(
			data, this->primary_base_partial);

		return this->warpscan[1][SrtsGrid::RAKING_THREADS - 1];
	}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ CollectiveScan(
		uint4 smem_pool[SrtsGrid::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)]) :
			reduction::CollectiveReduction<SrtsGrid>(smem_pool, warpscan) {}


	/**
	 * Initializer
	 */
	template <T Identity()>
	__device__ __forceinline__ void Initialize()
	{
		// Initialize warpscan
		if (threadIdx.x < B40C_WARP_THREADS(__B40C_CUDA_ARCH__)) {
			this->warpscan[0][threadIdx.x] = Identity();
		}
	}
};

/******************************************************************************
 * CollectiveScan Implementation (specialized for two-level SRTS grid)
 ******************************************************************************/


/**
 * Base class for tile-scanning routines (specialized for two-level SRTS grid)
 *
 * Extends CollectiveReduction for two-level SRTS grids
 */
template <typename SrtsGrid, typename SecondarySrtsGrid>
struct CollectiveScan : reduction::CollectiveReduction<SrtsGrid>
{
	typedef typename SrtsGrid::T T;

	/**
	 * Scan a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ void ScanTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(
			data, this->primary_base_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(
				this->primary_raking_seg);
			this->secondary_base_partial[0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<SecondarySrtsGrid, true, BinaryOp>(
			this->secondary_raking_seg, this->warpscan, carry);

		__syncthreads();

		// Raking scan in primary grid seeded by partial from secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = this->secondary_base_partial[0];
			SerialScan<T, SrtsGrid::PARTIALS_PER_SEG, true, BinaryOp>::Invoke(
				this->primary_raking_seg, partial);
		}

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<SrtsGrid, VEC_SIZE, EXCLUSIVE, BinaryOp>::Invoke(
			data, this->primary_base_partial);
	}


	/**
	 * Scan a single tile.  Inclusive aggregate is returned to all threads.
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T BinaryOp(const T&, const T&)>
	__device__ __forceinline__ T ScanTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<SrtsGrid, VEC_SIZE, BinaryOp>::Invoke(data, this->primary_base_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, BinaryOp>::Invoke(
				this->primary_raking_seg);
			this->secondary_base_partial[0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<SecondarySrtsGrid, true, BinaryOp>(
			this->secondary_raking_seg, this->warpscan);

		__syncthreads();

		// Raking scan in primary grid seeded by partial from secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = this->secondary_base_partial[0];
			SerialScan<T, SrtsGrid::PARTIALS_PER_SEG, true, BinaryOp>::Invoke(
				this->primary_raking_seg, partial);
		}

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<SrtsGrid, VEC_SIZE, EXCLUSIVE, BinaryOp>::Invoke(
			data, this->primary_base_partial);

		return this->warpscan[1][SrtsGrid::RAKING_THREADS - 1];
	}


	/**
	 * Constructor
	 */
	__device__ __forceinline__ CollectiveScan(
		uint4 smem_pool[SrtsGrid::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)]) :
			reduction::CollectiveReduction<SrtsGrid>(smem_pool, warpscan) {}


	/**
	 * Initializer
	 */
	template <T Identity()>
	__device__ __forceinline__ void Initialize()
	{
		// Initialize warpscan
		if (threadIdx.x < B40C_WARP_THREADS(__B40C_CUDA_ARCH__)) {
			this->warpscan[0][threadIdx.x] = Identity();
		}
	}
};


/******************************************************************************
 * Helper structures
 ******************************************************************************/

/**
 * Scan each load in registers, seeding from smem partials
 */
template <
	typename SrtsGrid,
	int VEC_SIZE,
	bool EXCLUSIVE,
	typename SrtsGrid::T BinaryOp(const typename SrtsGrid::T&, const typename SrtsGrid::T&)>
struct ScanVectors
{
	typedef typename SrtsGrid::T T;

	// Next load
	template <int LOAD, int TOTAL_LOADS>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
			T *base_partial)
		{
			T exclusive_partial = base_partial[LOAD * SrtsGrid::LANE_STRIDE];
			SerialScan<T, VEC_SIZE, EXCLUSIVE, BinaryOp>::Invoke(data[LOAD], exclusive_partial);

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


} // namespace scan
} // namespace b40c

