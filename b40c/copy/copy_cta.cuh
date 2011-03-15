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
 * CopyCta tile-processing functionality for copy kernels
 ******************************************************************************/

#pragma once

namespace b40c {
namespace copy {


/**
 * Derivation of KernelConfig that encapsulates tile-processing
 * routines
 */
template <typename KernelConfig>
struct CopyCta : KernelConfig
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::T T;
	typedef typename KernelConfig::SizeT SizeT;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Input and output device pointers
	T* d_in;
	T* d_out;

	// Tile of elements
	T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ CopyCta(T *d_in, T *d_out) :
		d_in(d_in), d_out(d_out) {}


	/**
	 * Process a single tile
	 *
	 * Each thread copys only the strided values it loads.
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		// Load tile
		util::LoadTile<
			T,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			UNGUARDED_IO>::Invoke(data, d_in, cta_offset, out_of_bounds);

		__syncthreads();

		// Store tile
		util::StoreTile<
			T,
			SizeT,
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			UNGUARDED_IO>::Invoke(data, d_out, cta_offset, out_of_bounds);
	}
};



} // namespace copy
} // namespace b40c

