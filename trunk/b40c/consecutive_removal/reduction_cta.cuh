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
#include <b40c/util/reduction/operators.cuh>

namespace b40c {
namespace reduction {


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig>
struct ReductionCta : KernelConfig
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::T T;							// Input type for detecting consecutive discontinuities
	typedef typename KernelConfig::SizeT SizeT;					// Type for counting discontinuities

	// Tuple of (input, discontinuity-count)
	typedef typename util::Tuple<T, SizeT> SoaTuple;

	// SOA of (input) tuples
	typedef util::Tuple<T (*)[KernelConfig::LOAD_VEC_SIZE]> DataSoa;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Accumulator for the number of discontinuities observed (in each thread)
	SizeT carry;

	// Input device pointer
	T* d_in;

	// Output spine pointer
	SizeT* d_spine;

	// Tile of elements
	T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

	// Smem storage for discontinuity-count reduction tree
	uint4 (&reduction_tree)[KernelConfig::SMEM_QUADS];


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * SOA input-equivalence reduction operator
	 */
	static __device__ __forceinline__ SoaTuple SoaReductionOp(
		SoaTuple &first,
		SoaTuple &second)
	{
		return (first.t0 == second.t0) ?
			first :									// Inputs are same: keep first's value and count
			SoaTuple(second.t0, first.t1 + 1);		// Inputs are different: keep second's value with first's count incremented
	}

	template <bool FIRST_TILE>
	struct ReduceHeadFlags
	{
		// Next load
		template <int LOAD, int TOTAL_LOADS>
		struct Iterate
		{
			static __device__ __forceinline__ void Invoke(
				ReductionCta *reduction_cta,
				SizeT cta_offset)
			{
				// Compute the exclusive discontinuity-tuple to use for reducing
				// discontinuities in this load
				SoaTuple discontinuity_tuple;

				if (!FIRST_TILE || (threadIdx.x > 0) || (blockIdx.x > 0)) {

					// Retrieve previous thread's last vector element
					util::ModifiedLoad<T, KernelConfig::READ_MODIFIER>::Ld(
						discontinuity_tuple.t0,
						reduction_cta->d_in,
						cta_offset + (LOAD * LOAD_STRIDE) + (threadIdx.x << KernelConfig::LOG_LOAD_VEC_SIZE) - 1);

					discontinuity_tuple.t1 = reduction_cta->carry;

				} else {

					// First load of first tile of first CTA: use same value as our first vec-element
					discontinuity_tuple.t0 = reduction_cta->data[0][0];
					discontinuity_tuple.t1 = 1;				// The first value counts as a discontinuity
				}

				// Reduce discontinuities into carry
				discontinuity_tuple = util::reduction::soa::SerialSoaReduceLane<
					SoaTuple,
					DataSoa,
					LOAD,
					KernelConfig::LOAD_VEC_SIZE,
					SoaReductionOp>::Invoke(
						DataSoa(reduction_cta->data),
						discontinuity_tuple);

				reduction_cta->carry = discontinuity_tuple.t1;

				// Next load
				Iterate<LOAD + 1, TOTAL_LOADS>::Invoke(reduction_cta, cta_offset);
			}
		};

		// Terminate
		template <int TOTAL_LOADS>
		struct Iterate<TOTAL_LOADS, TOTAL_LOADS>
		{
			static __device__ __forceinline__ void Invoke(
				ReductionCta *reduction_cta,
				SizeT cta_offset) {}
		};

		// Interface
		static __device__ __forceinline__ void Invoke(
			ReductionCta *reduction_cta,
			SizeT cta_offset)
		{
			Iterate<0, KernelConfig::LOADS_PER_TILE>::Invoke(reduction_cta, cta_offset);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	__device__ __forceinline__ ReductionCta(
		uint4 (&reduction_tree)[KernelConfig::SMEM_QUADS],
		T *d_in,
		SizeT *d_spine) :

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
		// Load tile
		util::LoadTile<
			T,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::Invoke(data, d_in, cta_offset, 0);

		ReduceHeadFlags<FIRST_TILE>::Invoke(this, cta_offset);

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
			util::reduction::DefaultSum<SizeT>,
			false,											// No need to return aggregate reduction in all threads
			true>::Invoke(									// All carry values are valid (i.e., they have been assigned at least once)
				carry,
				reinterpret_cast<T*>(reduction_tree),
				num_elements);

		// Write output
		if (threadIdx.x == 0) {
			printf("Reduced %d discontinuities\n", carry);
			util::ModifiedStore<T, ReductionCta::WRITE_MODIFIER>::St(
				carry, d_spine, blockIdx.x);
		}
	}
};



} // namespace reduction
} // namespace b40c

