/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
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
 * CTA-processing functionality for copy kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace copy {



/**
 * Templated texture reference for global input
 */
template <typename TexRefT>
struct InputTex
{
	static texture<TexRefT, cudaTextureType1D, cudaReadModeElementType> ref;
};
template <typename TexRefT>
texture<TexRefT, cudaTextureType1D, cudaReadModeElementType> InputTex<TexRefT>::ref;


/**
 * Copy CTA
 */
template <typename KernelPolicy>
struct Cta : KernelPolicy
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T T;
	typedef typename KernelPolicy::SizeT SizeT;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	T* d_in;
	T* d_out;

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		T *d_in,
		T *d_out) :
			d_in(d_in),
			d_out(d_out) {}


	/**
	 * Process a single full tile
	 *
	 * Each thread copies only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessFullTile(SizeT cta_offset)
	{
		// Tile of elements
		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadTileUnguarded(
				data,
				InputTex<TexRefT>::ref,
				d_in,
				cta_offset);

		__syncthreads();

		// Store tile
		util::io::StoreTile<
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Store(
				data,
				d_out,
				cta_offset,
				guarded_elements);
	}


	/**
	 * Process a single partially-full tile
	 *
	 * Each thread copies only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		// Tile of elements
		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::LoadValid(
				data,
				d_in,
				cta_offset,
				guarded_elements);

		__syncthreads();

		// Store tile
		util::io::StoreTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			KernelPolicy::CHECK_ALIGNMENT>::Store(
				data,
				d_out,
				cta_offset,
				guarded_elements);
	}
};



} // namespace copy
} // namespace b40c

