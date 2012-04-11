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
template <typename TexVec>
struct InputTex
{
	static texture<TexVec, cudaTextureType1D, cudaReadModeElementType> d_in_ref;
};
template <typename TexVec>
typename texture<TexVec, cudaTextureType1D, cudaReadModeElementType> InputTex<TexVec>::d_in_ref;


/**
 * Copy CTA
 */
template <typename KernelPolicy>
struct Cta : KernelPolicy
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::T 			T;					// Data type to reduce
	typedef typename KernelPolicy::TexVec		TexVec;				// Texture vector type
	typedef typename KernelPolicy::TexRef		TexRef;				// Texture reference type
	typedef typename KernelPolicy::SizeT 		SizeT;				// Counting type

	// Tile reader type
	typedef util::io::TileReader<
		KernelPolicy::THREADS,
		KernelPolicy::READ_MODIFIER> TileReader;

	// Tile writer type
	typedef util::io::TileWriter<
		KernelPolicy::THREADS,
		KernelPolicy::WRITE_MODIFIER> TileWriter;


	//---------------------------------------------------------------------
	// Constants
	//---------------------------------------------------------------------

	enum {
		CUDA_ARCH				= __B40C_CUDA_ARCH__,
		THREAD_OCCUPANCY		= B40C_SM_THREADS(CUDA_ARCH) >> KernelPolicy::LOG_THREADS,
		MAX_CTA_OCCUPANCY  		= B40C_MIN(B40C_SM_CTAS(CUDA_ARCH), THREAD_OCCUPANCY),
		VALID					= (MAX_CTA_OCCUPANCY > 0),
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	T* 					d_in;				// Input device pointer
	T* 					d_out;				// Output device pointer
	TexRef 				d_in_ref;			// Input texture reference

	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		T *d_in,
		T *d_out) :
			// Initializers
			d_in(d_in),
			d_out(d_out),
			d_in_ref(InputTex<TexVec>::d_in_ref)
	{
	}


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
		TileReader::LoadUnguarded(data, d_in_ref, d_in, cta_offset);

		__syncthreads();

		// Store tile
		TileWriter::StoreUnguarded(data, d_out, cta_offset);
	}


	/**
	 * Process a single partially-full tile
	 *
	 * Each thread copies only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT guarded_elements)
	{
		// Tile of elements
		T data[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];

		// Load tile
		TileReader::LoadGuarded(data, d_in, cta_offset, guarded_elements);

		__syncthreads();

		// Store tile
		TileWriter::StoreGuarded(data, d_out, cta_offset, guarded_elements);
	}
};



} // namespace copy
} // namespace b40c

