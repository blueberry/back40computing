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
 * Tile-processing functionality for consecutive-removal upsweep
 * reduction kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>
#include <b40c/util/operators.cuh>
#include <b40c/util/io/load_tile.cuh>

namespace b40c {
namespace reduction {


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig>
struct UpsweepCta : KernelConfig
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::SizeT 		SizeT;
	typedef typename KernelConfig::T 			T;					// Input type for detecting consecutive discontinuities
	typedef typename KernelConfig::FlagCount 	FlagCount;			// Type for counting discontinuities
	typedef int 								LocalFlagCount;		// Type for local discontinuity counts (just needs to count up to TILE_ELEMENTS_PER_THREAD)

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Accumulator for the number of discontinuities observed (in each thread)
	FlagCount carry;

	// Input device pointer
	T* d_in;

	// Output spine pointer
	FlagCount* d_spine;

	// Smem storage for discontinuity-count reduction tree
	uint4 (&reduction_tree)[KernelConfig::SMEM_QUADS];


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	__device__ __forceinline__ UpsweepCta(
		uint4 (&reduction_tree)[KernelConfig::SMEM_QUADS],
		T *d_in,
		FlagCount *d_spine) :

			reduction_tree(reduction_tree),
			d_in(d_in),
			d_spine(d_spine),
			carry(0) {}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{

		T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];					// Tile of elements
		LocalFlagCount flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];	// Tile of discontinuity flags

		// Load data tile, initializing discontinuity flags
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::template Invoke<FIRST_TILE>(							// Full-tile == unguarded loads
				data, flags, d_in, cta_offset);

		// Reduce flags, accumulate in carry
		carry += util::reduction::SerialReduce<
			LocalFlagCount,
			KernelConfig::TILE_ELEMENTS_PER_THREAD>::Invoke((LocalFlagCount*) flags);

		__syncthreads();
	}


	/**
	 * Collective reduction across all threads, stores final reduction to output
	 *
	 * Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		// Cooperatively reduce the carries in each thread (thread-0 gets the result)
		carry = util::reduction::TreeReduce<
			LocalFlagCount,
			KernelConfig::LOG_THREADS,
			util::DefaultSum>::Invoke<false>(				// No need to return aggregate reduction in all threads
				carry,
				(LocalFlagCount*) reduction_tree);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<UpsweepCta::WRITE_MODIFIER>::St(
				carry, d_spine + blockIdx.x);
		}
	}
};



} // namespace reduction
} // namespace b40c

