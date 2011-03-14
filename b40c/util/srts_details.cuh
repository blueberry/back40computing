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
 * Operational details for threads working in an SRTS grid
 ******************************************************************************/

#pragma once

namespace b40c {
namespace util {


/**
 * Operational details for threads working in an SRTS grid
 */
template <
	typename SrtsGrid,
	typename SrtsGrid::T Identity(),
	typename SecondarySrtsGrid = typename SrtsGrid::SecondaryGrid>
struct SrtsDetails;


/**
 * Operational details for threads working in an SRTS grid (specialized for one-level SRTS grid)
 */
template <
	typename SrtsGrid,
	typename SrtsGrid::T Identity()>
struct SrtsDetails<SrtsGrid, Identity, NullType> : SrtsGrid
{
	typedef typename SrtsGrid::T T;
	typedef T WarpscanStorage [2][B40C_WARP_THREADS(SrtsGrid::CUDA_ARCH)];
	typedef NullType SecondarySrtsDetails;

	enum {
		CUMULATIVE_THREAD 				= SrtsGrid::RAKING_THREADS - 1
	};

	/**
	 * Warpscan storage
	 */
	WarpscanStorage &warpscan;

	/**
	 * The location in the smem grid where the calling thread can insert/extract
	 * its partial for raking reduction/scan into the first lane.
	 */
	typename SrtsGrid::LanePartial lane_partial;

	/**
	 * Returns the location in the smem grid where the calling thread can begin serial
	 * raking/scanning
	 */
	typename SrtsGrid::RakingSegment raking_segment;

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ SrtsDetails(
		uint4 *smem_pool,
		WarpscanStorage &warpscan) :
			warpscan(warpscan),
			lane_partial(SrtsGrid::MyLanePartial(smem_pool))						// set lane partial pointer
	{
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {

			if (Identity != NULL) {
				// Initialize first half of warpscan storage to identity
				warpscan[0][threadIdx.x] = Identity();
			}

			// Set raking segment pointer
			raking_segment = SrtsGrid::MyRakingSegment(smem_pool);
		}
	}

	/**
	 * Return the cumulative partial left in the final warpscan cell
	 */
	__device__ __forceinline__ T CumulativePartial() const
	{
		return warpscan[1][CUMULATIVE_THREAD];
	}
};


/**
 * Operational details for threads working in a hierarchical SRTS grid
 */
template <
	typename SrtsGrid,
	typename SrtsGrid::T Identity(),
	typename SecondarySrtsGrid>
struct SrtsDetails : SrtsGrid
{
	typedef typename SrtsGrid::T T;
	typedef T WarpscanStorage [2][B40C_WARP_THREADS(SrtsGrid::CUDA_ARCH)];
	typedef SrtsDetails<SecondarySrtsGrid, Identity> SecondarySrtsDetails;

	enum {
		CUMULATIVE_THREAD = SecondarySrtsDetails::CUMULATIVE_THREAD
	};

	/**
	 * The location in the smem grid where the calling thread can insert/extract
	 * its partial for raking reduction/scan into the first lane.
	 */
	typename SrtsGrid::LanePartial lane_partial;

	/**
	 * Returns the location in the smem grid where the calling thread can begin serial
	 * raking/scanning
	 */
	typename SrtsGrid::RakingSegment raking_segment;

	/**
	 * Secondary-level grid details
	 */
	SecondarySrtsDetails secondary_details;

	/**
	 * Constructor
	 */
	__host__ __device__ __forceinline__ SrtsDetails(
		uint4 *smem_pool,
		WarpscanStorage &warpscan) :
			lane_partial(SrtsGrid::MyLanePartial(smem_pool)),							// set lane partial pointer
			secondary_details(smem_pool + SrtsGrid::PRIMARY_SMEM_QUADS, warpscan)
	{
		if (threadIdx.x < SrtsGrid::RAKING_THREADS) {
			// Set raking segment pointer
			raking_segment = SrtsGrid::MyRakingSegment(smem_pool);
		}
	}

	/**
	 * Return the cumulative partial left in the final warpscan cell
	 */
	__device__ __forceinline__ T CumulativePartial() const
	{
		return secondary_details.CumulativePartial();
	}
};





} // namespace util
} // namespace b40c

