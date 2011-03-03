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
 * SRTS Grid Description
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace util {


/**
 * An invalid SRTS grid type
 */
struct InvalidSrtsGrid
{
	enum { SMEM_QUADS = 0 };
};


/**
 * Description of a (typically) conflict-free serial-reduce-then-scan (SRTS) 
 * shared-memory grid.
 *
 * A "lane" for reduction/scan consists of one value (i.e., "partial") per
 * active thread.  A grid consists of one or more scan lanes. The lane(s)
 * can be sequentially "raked" by the specified number of raking threads
 * (e.g., for upsweep reduction or downsweep scanning), where each raking
 * thread progresses serially through a segment that is its share of the
 * total grid.
 *
 * Depending on how the raking threads are further reduced/scanned, the lanes
 * can be independent (i.e., only reducing the results from every
 * SEGS_PER_LANE raking threads), or fully dependent (i.e., reducing the
 * results from every raking thread)
 */
template <
	typename _T,							// Type of items we will be reducing/scanning
	int _LOG_ACTIVE_THREADS, 						// Number of threads placing a lane partial (i.e., the number of partials per lane)
	int _LOG_SCAN_LANES,							// Number of scan lanes
	int _LOG_RAKING_THREADS, 						// Number of threads used for raking (typically 1 warp)
	typename SecondarySrtsGrid = InvalidSrtsGrid>	// Whether or not the application calls for a two-level grid (e.g., dependent scan lanes > arch warp threads)
struct SrtsGrid
{
	// Type of items we will be reducing/scanning
	typedef _T T;
	
	// Secondary SRTS grid type (if specified)
	typedef SecondarySrtsGrid SecondaryGrid;

	// N.B.: We use an enum type here b/c of a NVCC-win compiler bug where the
	// compiler can't handle ternary expressions in static-const fields having
	// both evaluation targets as local const expressions.
	enum {

		// Number number of partials per lane
		LOG_PARTIALS_PER_LANE 			= _LOG_ACTIVE_THREADS,
		PARTIALS_PER_LANE				= 1 << LOG_PARTIALS_PER_LANE,

		// Number of scan lanes
		LOG_SCAN_LANES					= _LOG_SCAN_LANES,
		SCAN_LANES						= 1 <<LOG_SCAN_LANES,

		// Number of raking threads
		LOG_RAKING_THREADS				= _LOG_RAKING_THREADS,
		RAKING_THREADS					= 1 << LOG_RAKING_THREADS,

		// Total number of partials in the grid (all lanes)
		LOG_PARTIALS					= LOG_PARTIALS_PER_LANE + LOG_SCAN_LANES,
		PARTIALS			 			= 1 << LOG_PARTIALS,

		// Partials to be raked per raking thread
		LOG_PARTIALS_PER_SEG 			= LOG_PARTIALS - LOG_RAKING_THREADS,
		PARTIALS_PER_SEG 				= 1 << LOG_PARTIALS_PER_SEG,
	
		// Number of partials that we can put in one stripe across the shared memory banks
		LOG_PARTIALS_PER_BANK_ARRAY		= B40C_LOG_MEM_BANKS(__B40C_CUDA_ARCH__) +
											B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) -
											Log2<sizeof(T)>::VALUE,
	
		// Number of partials that we must use to "pad out" one memory bank
		LOG_PADDING_PARTIALS			= B40C_MAX(0, B40C_LOG_BANK_STRIDE_BYTES(__B40C_CUDA_ARCH__) - Log2<sizeof(T)>::VALUE),
		PADDING_PARTIALS				= 1 << LOG_PADDING_PARTIALS,
	
		// Number of consecutive partials we can have without padding (i.e., a "row")
		LOG_PARTIALS_PER_ROW			= B40C_MAX(LOG_PARTIALS_PER_SEG, LOG_PARTIALS_PER_BANK_ARRAY),
		PARTIALS_PER_ROW				= 1 << LOG_PARTIALS_PER_ROW,

		// Number of partials (including padding) per "row"
		PADDED_PARTIALS_PER_ROW			= PARTIALS_PER_ROW + PADDING_PARTIALS,

		// Number of raking segments per row (i.e., number of raking threads per row)
		LOG_SEGS_PER_ROW 				= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG,
		SEGS_PER_ROW					= 1 << LOG_SEGS_PER_ROW,
	
		// Number of rows in the grid
		LOG_ROWS						= B40C_MAX(0, LOG_PARTIALS - LOG_PARTIALS_PER_ROW),
		ROWS 							= 1 << LOG_ROWS,
	
		// Number of rows per lane
		LOG_ROWS_PER_LANE				= B40C_MAX(0, LOG_ROWS - LOG_SCAN_LANES),
		ROWS_PER_LANE					= 1 << LOG_ROWS_PER_LANE,

		// Number of raking thraeds per lane
		LOG_RAKING_THREADS_PER_LANE		= LOG_SEGS_PER_ROW + LOG_ROWS_PER_LANE,
		RAKING_THREADS_PER_LANE			= 1 << LOG_RAKING_THREADS_PER_LANE,

		// Stride between lanes (in partials)
		LANE_STRIDE						= ROWS_PER_LANE * PADDED_PARTIALS_PER_ROW,

		// Total number of quad words (uint4) needed to back the grid
		SMEM_QUADS						= (((ROWS * PADDED_PARTIALS_PER_ROW * sizeof(T)) + sizeof(uint4) - 1) / sizeof(uint4)) +
											SecondarySrtsGrid::SMEM_QUADS
	};
	
	
	/**
	 * Returns the location in the smem grid where the calling thread can insert/extract
	 * its partial for raking reduction/scan into the first lane.  Positions in subsequent
	 * lanes can be obtained via increments of LANE_STRIDE.
	 */
	static __device__ __forceinline__ T* BasePartial(T *smem)
	{
		int row = threadIdx.x >> LOG_PARTIALS_PER_ROW;		
		int col = threadIdx.x & (PARTIALS_PER_ROW - 1);			
		return smem + (row * PADDED_PARTIALS_PER_ROW) + col;
	}
	
	/**
	 * Returns the location in the smem grid where the calling thread can begin serial
	 * raking/scanning
	 */
	static __device__ __forceinline__ T* RakingSegment(T *smem)
	{
		int row = threadIdx.x >> LOG_SEGS_PER_ROW;
		int col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		
		return smem + (row * PADDED_PARTIALS_PER_ROW) + col;
	}
};


} // namespace util
} // namespace b40c

