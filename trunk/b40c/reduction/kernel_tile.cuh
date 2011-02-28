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
 * Derivation of ReductionKernelConfig that encapsulates tile-processing routines
 ******************************************************************************/

#pragma once

#include <b40c/reduction/reduction_utils.cuh>

namespace b40c {
namespace reduction {


/******************************************************************************
 * ReductionTile Declaration
 ******************************************************************************/

/**
 * Derivation of ReductionKernelConfig that encapsulates tile-processing routines
 */
template <typename ReductionKernelConfig>
struct ReductionTile : ReductionKernelConfig
{
	typedef typename ReductionKernelConfig::T T;
	typedef typename ReductionKernelConfig::SizeT SizeT;

	/**
	 * Process a single tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool UNGUARDED_IO>
	static __device__ __forceinline__ void ProcessTile(
		T * __restrict 		d_in,
		SizeT 				cta_offset,
		SizeT 				out_of_bounds,
		T 					&carry);


	/**
	 * Load transform function for assigning identity to tile values
	 * that are out of range.
	 */
	static __device__ __forceinline__ void LoadTransform(
		T &val,
		bool in_bounds);


	/**
	 * Collective reduction across all threads
	 *
	 * Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 */
	static __device__ __forceinline__ void CollectiveReduction(
		T carry,
		T *d_out);

};



/******************************************************************************
 * ReductionTile Implementation
 ******************************************************************************/

/**
 * Process a single tile
 */
template <typename ReductionKernelConfig>
template <bool UNGUARDED_IO>
void ReductionTile<ReductionKernelConfig>::ProcessTile(
	T * __restrict 		d_in,
	SizeT 				cta_offset,
	SizeT 				out_of_bounds,
	T 					&carry)					// in/out param
{
	T data[LOADS_PER_TILE][LOAD_VEC_SIZE];

	// Load tile
	util::LoadTile<
		T,
		SizeT,
		LOG_LOADS_PER_TILE,
		LOG_LOAD_VEC_SIZE,
		THREADS,
		READ_MODIFIER,
		UNGUARDED_IO,
		LoadTransform>::Invoke(data, d_in, cta_offset, out_of_bounds);

	// Reduce the data we loaded for this tile
	T tile_partial = SerialReduce<T, TILE_ELEMENTS_PER_THREAD, BinaryOp>::Invoke(
		reinterpret_cast<T*>(data));

	// Reduce into carry
	carry = BinaryOp(carry, tile_partial);

	__syncthreads();
}


/**
 * Load transform function for assigning identity to tile values
 * that are out of range.
 */
template <typename ReductionKernelConfig>
void ReductionTile<ReductionKernelConfig>::LoadTransform(
	T &val,
	bool in_bounds)
{
	// Assigns identity value to out-of-bounds loads
	if (!in_bounds) val = Identity();
}


/**
 * Collective reduction across all threads: One-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have one warp or smaller of raking threads.
 */
template <typename ReductionTile, bool TWO_LEVEL_GRID>
struct CollectiveReductionHelper
{
	typedef typename ReductionTile::T T;

	static __device__ __forceinline__ void Invoke(T carry, T *d_out)
	{
		// Shared memory pool
		__shared__ uint4 smem_pool[ReductionTile::SMEM_QUADS];

		T *primary_grid = reinterpret_cast<T*>(smem_pool);

		CollectiveReduction<T, ReductionTile::PrimaryGrid, ReductionTile::BinaryOp, ReductionTile::WRITE_MODIFIER>(
			carry, d_out, primary_grid);
	}
};


/**
 * Collective reduction across all threads: Two-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have more than one warp of raking threads.
 */
template <typename ReductionTile>
struct CollectiveReductionHelper <ReductionTile, true>
{
	typedef typename ReductionTile::T T;

	static __device__ __forceinline__ void Invoke(T carry, T *d_out)
	{
		// Shared memory pool
		__shared__ uint4 smem_pool[ReductionTile::SMEM_QUADS];

		T *primary_grid = reinterpret_cast<T*>(smem_pool);
		T *secondary_grid = reinterpret_cast<T*>(smem_pool + ReductionTile::PrimaryGrid::SMEM_QUADS);		// Offset by the primary grid

		CollectiveReduction<T, ReductionTile::PrimaryGrid, ReductionTile::BinaryOp, ReductionTile::WRITE_MODIFIER>(
			carry, d_out, primary_grid, secondary_grid);
	}
};


/**
 * Collective reduction across all threads
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.
 */
template <typename ReductionKernelConfig>
void ReductionTile<ReductionKernelConfig>::CollectiveReduction(
	T carry,
	T *d_out)
{
	CollectiveReductionHelper<ReductionTile, TWO_LEVEL_GRID>::Invoke(carry, d_out);
}




} // namespace reduction
} // namespace b40c

