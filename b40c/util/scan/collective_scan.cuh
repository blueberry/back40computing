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
#include <b40c/util/reduction/collective_reduction.cuh>
#include <b40c/util/scan/serial_scan.cuh>
#include <b40c/util/scan/warp_rake_and_scan.cuh>

namespace b40c {
namespace util {
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
	typename T,
	T ScanOp(const T&, const T&),
	typename LanePartial,
	int SCAN_LANES,
	int VEC_SIZE,
	bool EXCLUSIVE>
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
	typedef typename SrtsGrid::LanePartial PrimaryLanePartial;

	/**
	 * Scan a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T ScanOp(const T&, const T&)>
	__device__ __forceinline__ void ScanTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, this->primary_lane_partial);

		__syncthreads();

		// Primary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<T, ScanOp, SrtsGrid::PARTIALS_PER_SEG, SrtsGrid::LOG_RAKING_THREADS, true>(
			this->primary_raking_seg, this->warpscan, carry);

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE, EXCLUSIVE>::Invoke(
			data, this->primary_lane_partial);
	}


	/**
	 * Scan a single tile.  Total reduction is returned to all threads
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T ScanOp(const T&, const T&)>
	__device__ __forceinline__ T ScanTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, this->primary_lane_partial);

		__syncthreads();

		// Primary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<T, ScanOp, SrtsGrid::PARTIALS_PER_SEG, SrtsGrid::LOG_RAKING_THREADS, true>(
			this->primary_raking_seg, this->warpscan);

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE, EXCLUSIVE>::Invoke(
			data, this->primary_lane_partial);

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
	typedef typename SrtsGrid::LanePartial PrimaryLanePartial;
	typedef typename SecondarySrtsGrid::LanePartial SecondaryLanePartial;

	/**
	 * Scan a single tile.  Carry-in/out is seeded/updated only in raking threads (homogeneously)
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T ScanOp(const T&, const T&)>
	__device__ __forceinline__ void ScanTileWithCarry(
		T data[SrtsGrid::SCAN_LANES][VEC_SIZE],
		T &carry)
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, this->primary_lane_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

			T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, ScanOp>::Invoke(
				this->primary_raking_seg);
			this->secondary_lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<T, ScanOp, SecondarySrtsGrid::PARTIALS_PER_SEG, SecondarySrtsGrid::LOG_RAKING_THREADS, true>(
			this->secondary_raking_seg, this->warpscan, carry);

		__syncthreads();

		// Raking scan in primary grid seeded by partial from secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = this->secondary_lane_partial[0][0];
			SerialScan<T, SrtsGrid::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				this->primary_raking_seg, partial);
		}

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE, EXCLUSIVE>::Invoke(
			data, this->primary_lane_partial);
	}


	/**
	 * Scan a single tile.  Inclusive aggregate is returned to all threads.
	 */
	template <
		int VEC_SIZE,
		bool EXCLUSIVE,
		T ScanOp(const T&, const T&)>
	__device__ __forceinline__ T ScanTile(T data[SrtsGrid::SCAN_LANES][VEC_SIZE])
	{
		// Reduce in registers, place partials in smem
		reduction::ReduceVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE>::Invoke(
			data, this->primary_lane_partial);

		__syncthreads();

		// Raking reduction in primary grid, place result partial into secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

			T partial = reduction::SerialReduce<T, SrtsGrid::PARTIALS_PER_SEG, ScanOp>::Invoke(
				this->primary_raking_seg);
			this->secondary_lane_partial[0][0] = partial;
		}

		__syncthreads();

		// Secondary rake and scan (guaranteed one warp or fewer raking threads)
		WarpRakeAndScan<T, ScanOp, SecondarySrtsGrid::PARTIALS_PER_SEG, SecondarySrtsGrid::LOG_RAKING_THREADS, true>(
			this->secondary_raking_seg, this->warpscan);

		__syncthreads();

		// Raking scan in primary grid seeded by partial from secondary grid
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			T partial = this->secondary_lane_partial[0][0];
			SerialScan<T, SrtsGrid::PARTIALS_PER_SEG, true, ScanOp>::Invoke(
				this->primary_raking_seg, partial);
		}

		__syncthreads();

		// Extract partials from smem, scan in registers
		ScanVectors<T, ScanOp, PrimaryLanePartial, SrtsGrid::SCAN_LANES, VEC_SIZE, EXCLUSIVE>::Invoke(
			data, this->primary_lane_partial);

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
	typename T,
	T ScanOp(const T&, const T&),
	typename LanePartial,
	int SCAN_LANES,
	int VEC_SIZE,
	bool EXCLUSIVE>
struct ScanVectors
{
	// Next lane
	template <int LANE, int TOTAL_LANES>
	struct Iterate {
		static __device__ __forceinline__ void Invoke(
			T data[SCAN_LANES][VEC_SIZE],
			LanePartial lane_partial)
		{
			T exclusive_partial = lane_partial[LANE][0];
			SerialScan<T, VEC_SIZE, EXCLUSIVE, ScanOp>::Invoke(data[LANE], exclusive_partial);

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

} // namespace scan
} // namespace util
} // namespace b40c

