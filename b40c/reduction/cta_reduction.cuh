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
#include <b40c/reduction/cta_reduction_base.cuh>

namespace b40c {
namespace reduction {


/******************************************************************************
 * CtaReduction Declaration
 ******************************************************************************/

/**
 * Derivation of ReductionKernelConfig that encapsulates tile-processing
 * routines
 */
template <typename ReductionKernelConfig>
struct CtaReduction :
	ReductionKernelConfig,
	CtaReductionBase<typename ReductionKernelConfig::SrtsGrid>
{
	typedef typename ReductionKernelConfig::T T;
	typedef typename ReductionKernelConfig::SizeT SizeT;

	// The value we will accumulate (in each thread)
	T carry;

	// Input and output device pointers
	T* d_in;
	T* d_out;

	// Tile of elements
	T data[ReductionKernelConfig::LOADS_PER_TILE][ReductionKernelConfig::LOAD_VEC_SIZE];

	/**
	 * Process a single tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds);


	/**
	 * Load transform function for assigning identity to tile values
	 * that are out of range.
	 */
	static __device__ __forceinline__ void LoadTransform(
		T &val,
		bool in_bounds);


	/**
	 * Collective reduction across all threads, stores final reduction to output
	 *
	 * Used to collectively reduce each thread's aggregate after striding through
	 * the input.
	 */
	__device__ __forceinline__ void FinalReduction();


	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaReduction(
		uint4 smem_pool[ReductionKernelConfig::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)],
		T *d_in,
		T *d_out) :
			CtaReductionBase<typename ReductionKernelConfig::SrtsGrid>(smem_pool, warpscan),
			carry(CtaReduction::Identity()),
			d_in(d_in),
			d_out(d_out) {}
};


/******************************************************************************
 * CtaReduction Implementation
 ******************************************************************************/

/**
 * Process a single tile
 */
template <typename ReductionKernelConfig>
template <bool UNGUARDED_IO>
void CtaReduction<ReductionKernelConfig>::ProcessTile(
	SizeT cta_offset,
	SizeT out_of_bounds)
{
	// Load tile
	util::LoadTile<
		T,
		SizeT,
		CtaReduction::LOG_LOADS_PER_TILE,
		CtaReduction::LOG_LOAD_VEC_SIZE,
		CtaReduction::THREADS,
		CtaReduction::READ_MODIFIER,
		UNGUARDED_IO,
		LoadTransform>::Invoke(data, d_in, cta_offset, out_of_bounds);

	// Reduce the data we loaded for this tile
	T tile_partial = SerialReduce<T, CtaReduction::TILE_ELEMENTS_PER_THREAD, CtaReduction::BinaryOp>::Invoke(
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
void CtaReduction<ReductionKernelConfig>::LoadTransform(
	T &val,
	bool in_bounds)
{
	// Assigns identity value to out-of-bounds loads
	if (!in_bounds) val = CtaReduction::Identity();
}


/**
 * Collective reduction across all threads, stores final reduction to output
 */
template <typename ReductionKernelConfig>
void CtaReduction<ReductionKernelConfig>::FinalReduction()
{
	T total = this->template ReduceTile<1, CtaReduction::BinaryOp>(
		reinterpret_cast<T (*)[1]>(&carry));

	// Write output
	if (threadIdx.x == 0) {
		util::ModifiedStore<T, CtaReduction::WRITE_MODIFIER>::St(total, d_out, 0);
	}
}


} // namespace reduction
} // namespace b40c

