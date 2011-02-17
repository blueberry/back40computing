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
 ******************************************************************************/

/******************************************************************************
 * Memcopy kernel
 ******************************************************************************/

#pragma once

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_utils.cuh"
#include "b40c_kernel_data_movement.cuh"

namespace b40c {
namespace memcopy {


/******************************************************************************
 * Kernel Configuration  
 ******************************************************************************/

/**
 * A detailed memcopy configuration type that specializes kernel code for a specific
 * memcopy pass.  It encapsulates granularity details derived from the inherited
 * MemcopyConfigType
 */
template <typename MemcopyConfigType>
struct MemcopyKernelConfig : MemcopyConfigType
{
	static const int THREADS						= 1 << MemcopyConfigType::LOG_THREADS;

	static const int LOG_WARPS						= MemcopyConfigType::LOG_THREADS - B40C_LOG_WARP_THREADS(__B40C_CUDA_ARCH__);
	static const int WARPS							= 1 << LOG_WARPS;

	static const int LOAD_VEC_SIZE					= 1 << MemcopyConfigType::LOG_LOAD_VEC_SIZE;
	static const int LOADS_PER_TILE					= 1 << MemcopyConfigType::LOG_LOADS_PER_TILE;

	static const int LOG_TILE_ELEMENTS_PER_THREAD	= MemcopyConfigType::LOG_LOAD_VEC_SIZE + MemcopyConfigType::LOG_LOADS_PER_TILE;
	static const int TILE_ELEMENTS_PER_THREAD		= 1 << LOG_TILE_ELEMENTS_PER_THREAD;

	static const int LOG_TILE_ELEMENTS 				= LOG_TILE_ELEMENTS_PER_THREAD + MemcopyConfigType::LOG_THREADS;
	static const int TILE_ELEMENTS					= 1 << LOG_TILE_ELEMENTS;
};



/******************************************************************************
 * Memcopy kernel subroutines
 ******************************************************************************/


template <typename Config, bool UNGUARDED_IO>
__device__ __forceinline__ void ProcessTile(
	typename Config::T * __restrict d_out,
	typename Config::T * __restrict d_in,
	typename Config::IndexType 	cta_offset,
	typename Config::IndexType 	out_of_bounds)
{
	typedef typename Config::T T;
	typedef typename Config::IndexType IndexType;

	T data[Config::LOADS_PER_TILE][Config::LOAD_VEC_SIZE];

	// Load tile
	LoadTile<
		T,
		IndexType,
		Config::LOG_LOADS_PER_TILE,
		Config::LOG_LOAD_VEC_SIZE,
		Config::THREADS,
		Config::CACHE_MODIFIER,
		UNGUARDED_IO>::Invoke(data, d_in, cta_offset, out_of_bounds);

//	__syncthreads();

	// Store tile
	StoreTile<
		T,
		IndexType,
		Config::LOG_LOADS_PER_TILE,
		Config::LOG_LOAD_VEC_SIZE,
		Config::THREADS,
		Config::CACHE_MODIFIER,
		UNGUARDED_IO>::Invoke(data, d_out, cta_offset, out_of_bounds);
}


/**
 * Memcopy pass
 */
template <typename Config>
__device__ __forceinline__ void MemcopyPass(
	typename Config::T * __restrict &d_out,
	typename Config::T * __restrict &d_in,
	CtaWorkDistribution<typename Config::IndexType> &work_decomposition)
{
	typedef typename Config::IndexType IndexType;
	
	// Determine our threadblock's work range
	IndexType cta_offset;			// Offset at which this CTA begins processing
	IndexType cta_elements;			// Total number of elements for this CTA to process
	IndexType guarded_offset; 		// Offset of final, partially-full tile (requires guarded loads)
	IndexType guarded_elements;		// Number of elements in partially-full tile

	work_decomposition.GetCtaWorkLimits<Config::LOG_TILE_ELEMENTS, Config::LOG_SCHEDULE_GRANULARITY>(
		cta_offset, cta_elements, guarded_offset, guarded_elements);

	// Copy full tiles of tile_elements
	while (cta_offset < guarded_offset) {

		ProcessTile<Config, true>(d_out, d_in, cta_offset, cta_elements);
		cta_offset += Config::TILE_ELEMENTS;
	}

	// Clean up last partial tile with guarded-io
	if (guarded_elements) {

		ProcessTile<Config, false>(d_out, d_in, cta_offset, cta_elements);
	}
}


/**
 * Upsweep reduction kernel entry point 
 */
template <typename KernelConfig>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__ 
void MemcopyKernel(
	typename KernelConfig::T * __restrict d_out,
	typename KernelConfig::T * __restrict d_in,
	CtaWorkDistribution<typename KernelConfig::IndexType> work_decomposition)
{
	MemcopyPass<KernelConfig>(d_out, d_in, work_decomposition);
}


/**
 * Wrapper stub for arbitrary types to quiet the linker
 */
template <typename KernelConfig>
void __wrapper__device_stub_MemcopyKernel(
	typename KernelConfig::T * __restrict &,
	typename KernelConfig::T * __restrict &,
	CtaWorkDistribution<typename KernelConfig::IndexType> &) {}



} // namespace memcopy
} // namespace b40c

