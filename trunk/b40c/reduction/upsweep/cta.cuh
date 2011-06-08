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
 * Reduction CTA processing abstraction
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>

#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace reduction {
namespace upsweep {


/**
 * CTA
 */
template <typename KernelPolicy>
struct Cta : KernelPolicy
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 		T;
	typedef typename KernelPolicy::SizeT 	SizeT;
	typedef typename KernelPolicy::SmemStorage	SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// The value we will accumulate (in each thread)
	T carry;

	// Input and output device pointers
	T* &d_in;
	T* &d_out;

	// Smem storage for reduction tree
	T* reduction_tree;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		SmemStorage &smem_storage,
		T *&d_in,
		T *&d_out) :

			reduction_tree(smem_storage.smem_pool.reduction_tree),
			d_in(d_in),
			d_out(d_out) {}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		// Tile of elements
		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				data, d_in + cta_offset);

		// Reduce the data we loaded for this tile
		T tile_partial = util::reduction::SerialReduce<KernelPolicy::TILE_ELEMENTS_PER_THREAD>::template Invoke<
			T, KernelPolicy::BinaryOp>((T*) data);

		// Reduce into carry
		if (FIRST_TILE) {
			carry = tile_partial;
		} else {
			carry = BinaryOp(carry, tile_partial);
		}

		__syncthreads();
	}


	/**
	 * Process a single, partial tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FIRST_TILE>
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		T datum;
		cta_offset += threadIdx.x;

		if (FIRST_TILE) {
			if (cta_offset < out_of_bounds) {
				util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(carry, d_in + cta_offset);
				cta_offset += KernelPolicy::THREADS;
			}
		}

		// Process loads singly
		while (cta_offset < out_of_bounds) {
			util::io::ModifiedLoad<KernelPolicy::READ_MODIFIER>::Ld(datum, d_in + cta_offset);
			carry = KernelPolicy::BinaryOp(carry, datum);
			cta_offset += KernelPolicy::THREADS;
		}
	}


	/**
	 * Unguarded collective reduction across all threads, stores final reduction
	 * to output.  Used to collectively reduce each thread's aggregate after
	 * striding through the input.
	 *
	 * All threads assumed to have valid carry data.
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		carry = util::reduction::TreeReduce<
			T,
			KernelPolicy::LOG_THREADS,
			KernelPolicy::BinaryOp>::Invoke<false>( 		// No need to return aggregate reduction in all threads
				carry,
				(T*) reduction_tree);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_out + blockIdx.x);
		}
	}

	/**
	 * Guarded collective reduction across all threads, stores final reduction
	 * to output. Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 *
	 * Only threads with ranks less than num_elements are assumed to have valid
	 * carry data.
	 */
	__device__ __forceinline__ void OutputToSpine(int num_elements)
	{
		carry = util::reduction::TreeReduce<
			T,
			KernelPolicy::LOG_THREADS,
			KernelPolicy::BinaryOp>::Invoke<false>(			// No need to return aggregate reduction in all threads
				carry,
				reduction_tree,
				num_elements);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelPolicy::WRITE_MODIFIER>::St(
				carry, d_out + blockIdx.x);
		}
	}
};


} // namespace upsweep
} // namespace reduction
} // namespace b40c

