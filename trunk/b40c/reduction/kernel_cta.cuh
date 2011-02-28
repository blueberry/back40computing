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
 * ReductionCta Declaration
 ******************************************************************************/

/**
 * Derivation of ReductionKernelConfig that encapsulates tile-processing
 * routines
 */
template <
	typename ReductionKernelConfig,
	bool TWO_LEVEL_GRID = ReductionKernelConfig::TWO_LEVEL_GRID>
struct ReductionCta;


/**
 * Derivation of ReductionKernelConfig that encapsulates tile-processing
 * routines (one-level SRTS grid)
 */
template <typename ReductionKernelConfig>
struct ReductionCta<ReductionKernelConfig, false> : ReductionKernelConfig
{
	typedef typename ReductionKernelConfig::T T;
	typedef typename ReductionKernelConfig::SizeT SizeT;

	// The value we will accumulate
	T carry;
	T* d_in;
	T* d_out;

	/**
	 * Process a single tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT 				cta_offset,
		SizeT 				out_of_bounds);


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
	__device__ __forceinline__ void CollectiveReduction();


	/**
	 * Constructor
	 */
	__device__ __forceinline__ ReductionCta(
		T *d_in,
		T *d_out) :
			carry(Identity()),
			d_in(d_in),
			d_out(d_out)
	{}
};


/**
 * Derivation of ReductionKernelConfig that encapsulates tile-processing
 * routines (two-level SRTS grid)
 */
template <typename ReductionKernelConfig>
struct ReductionCta<ReductionKernelConfig, true> : ReductionCta<ReductionKernelConfig, false>
{
	typedef typename ReductionKernelConfig::T T;

	/**
	 * Collective reduction across all threads
	 *
	 * Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 */
	__device__ __forceinline__ void CollectiveReduction();

	/**
	 * Constructor
	 */
	__device__ __forceinline__ ReductionCta(
		T *d_in,
		T *d_out) : ReductionCta<ReductionKernelConfig, false>(d_in, d_out)
	{

	}

};



/******************************************************************************
 * ReductionCta Implementation
 ******************************************************************************/

/**
 * Process a single tile
 */
template <typename ReductionKernelConfig>
template <bool UNGUARDED_IO>
void ReductionCta<ReductionKernelConfig, false>::ProcessTile(
	SizeT cta_offset,
	SizeT out_of_bounds)
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
void ReductionCta<ReductionKernelConfig, false>::LoadTransform(
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
template <typename ReductionKernelConfig>
void ReductionCta<ReductionKernelConfig, false>::CollectiveReduction()
{
	// Shared memory pool
	__shared__ uint4 smem_pool[SMEM_QUADS];

	T *primary_grid = reinterpret_cast<T*>(smem_pool);

	b40c::reduction::CollectiveReduction<T, PrimaryGrid, BinaryOp, WRITE_MODIFIER>(
		carry, d_out, primary_grid);
}



/**
 * Collective reduction across all threads: Two-level raking grid
 *
 * Used to collectively reduce each thread's aggregate after striding through
 * the input.  For when we have more than one warp of raking threads.
 */
template <typename ReductionKernelConfig>
void ReductionCta<ReductionKernelConfig, true>::CollectiveReduction()
{
	// Shared memory pool
	__shared__ uint4 smem_pool[SMEM_QUADS];

	T *primary_grid = reinterpret_cast<T*>(smem_pool);
	T *secondary_grid = reinterpret_cast<T*>(smem_pool + PrimaryGrid::SMEM_QUADS);		// Offset by the primary grid

	b40c::reduction::CollectiveReduction<T, PrimaryGrid, SecondaryGrid, BinaryOp, WRITE_MODIFIER>(
		carry, d_out, primary_grid, secondary_grid);
}


} // namespace reduction
} // namespace b40c

