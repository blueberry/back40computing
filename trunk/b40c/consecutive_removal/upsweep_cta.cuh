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

#include <b40c/util/soa_tuple.cuh>
#include <b40c/util/reduction/soa/serial_soa_reduce.cuh>
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
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Reduces the discontinuity flags
	 */
	template <bool FIRST_TILE>
	struct ReduceFlags
	{
		// Next vector
		template <int LOAD, int VEC_ELEMENT, int TOTAL_VEC_ELEMENTS>
		struct IterateVecElement
		{
			static __device__ __forceinline__ FlagCount Invoke(
				T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE],
				T previous,
				FlagCount carry)
			{
				T current = data[LOAD][VEC_ELEMENT];
				carry += (previous == current);

				return IterateVecElement<LOAD, VEC_ELEMENT + 1, TOTAL_VEC_ELEMENTS>::Invoke(
					data, current, carry);
			}
		};

		// Terminate
		template <int LOAD, int TOTAL_VEC_ELEMENTS>
		struct IterateVecElement<LOAD, TOTAL_VEC_ELEMENTS, TOTAL_VEC_ELEMENTS>
		{
			static __device__ __forceinline__ FlagCount Invoke(
				T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE],
				T previous,
				FlagCount carry)
			{
				return carry;
			}
		};

		// Next load
		template <int LOAD, int TOTAL_LOADS>
		struct IterateLoad
		{
			static __device__ __forceinline__ FlagCount Invoke(
				T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE],
				T* d_in,
				SizeT cta_offset,
				FlagCount carry)
			{
				SizeT thread_offset = cta_offset + (LOAD * LOAD_STRIDE) +
					(threadIdx.x << KernelConfig::LOG_LOAD_VEC_SIZE) - 1;

				// Initialize rank of first vector element
				T current = data[LOAD][0];
				if (FIRST_TILE && (LOAD == 0) && (blockIdx.x == 0) && (threadIdx.x == 0)) {

					// First load of first tile of first CTA: discontinuity
					carry++;

				} else {

					// Retrieve previous thread's last vector element
					T previous;
					util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
						previous,
						d_in + thread_offset);

					carry += (previous == current);
				}

				// Initialize flags for remaining vector elements in this load
				IterateVecElement<LOAD, 1, KernelConfig::LOAD_VEC_SIZE>::Invoke(
					data, current, carry);

				// Next load
				return IterateLoad<LOAD + 1, TOTAL_LOADS>::Invoke(
					data, d_in, cta_offset, carry);
			}
		};

		// Terminate
		template <int TOTAL_LOADS>
		struct IterateLoad<TOTAL_LOADS, TOTAL_LOADS>
		{
			static __device__ __forceinline__ FlagCount Invoke(
				T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE],
				T* d_in,
				SizeT cta_offset,
				FlagCount carry)
			{
				return carry;
			}
		};

		// Interface
		static __device__ __forceinline__ FlagCount Invoke(
			T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE],
			T* d_in,
			SizeT cta_offset,
			FlagCount carry)
		{
			return IterateLoad<0, KernelConfig::LOADS_PER_TILE>::Invoke(
				data, d_in, cta_offset, carry);
		}
	};



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
		// Tile of elements
		T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::Invoke(data, d_in, cta_offset, 0);

		carry = ReduceFlags<FIRST_TILE>::Invoke(data, d_in, cta_offset, carry);

		__syncthreads();
	}


	/**
	 * Collective reduction across all threads, stores final reduction to output
	 *
	 * Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 */
	__device__ __forceinline__ void OutputToSpine(int num_elements)
	{
		carry = util::reduction::TreeReduce<
			T,
			KernelConfig::LOG_THREADS,
			util::DefaultSum<FlagCount>,
			false,											// No need to return aggregate reduction in all threads
			true>::Invoke(									// All carry values are valid (i.e., they have been assigned at least once)
				carry,
				reinterpret_cast<T*>(reduction_tree),
				num_elements);

		// Write output
		if (threadIdx.x == 0) {
//			printf("Reduced %d discontinuities\n", carry);
			util::io::ModifiedStore<UpsweepCta::WRITE_MODIFIER>::St(
				carry, d_spine + blockIdx.x);
		}
	}
};



} // namespace reduction
} // namespace b40c

