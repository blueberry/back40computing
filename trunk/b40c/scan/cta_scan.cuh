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
 * Derivation of ScanKernelConfig that encapsulates tile-processing routines
 ******************************************************************************/

#pragma once

#include <b40c/reduction/reduction_utils.cuh>
#include <b40c/scan/scan_utils.cuh>
#include <b40c/scan/cta_scan_base.cuh>

namespace b40c {
namespace scan {


/**
 * Derivation of ScanKernelConfig that encapsulates tile-processing
 * routines
 */
template <typename ScanKernelConfig>
struct CtaScan :
	ScanKernelConfig,
	CtaScanBase<typename ScanKernelConfig::SrtsGrid>
{
	typedef typename ScanKernelConfig::T T;
	typedef typename ScanKernelConfig::SizeT SizeT;

	// The value we will accumulate (in raking threads only)
	T carry;

	// Input and output device pointers
	T *d_in;
	T *d_out;

	// Tile of scan elements
	T data[ScanKernelConfig::LOADS_PER_TILE][ScanKernelConfig::LOAD_VEC_SIZE];

	/**
	 * Process a single tile
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{

		// Load tile
		util::LoadTile<T, SizeT, CtaScan::LOG_LOADS_PER_TILE, CtaScan::LOG_LOAD_VEC_SIZE, CtaScan::THREADS, CtaScan::READ_MODIFIER, UNGUARDED_IO>::Invoke(
			data, d_in, cta_offset, out_of_bounds);

		// Scan tile
		this->template ScanTileWithCarry<CtaScan::LOAD_VEC_SIZE, CtaScan::BinaryOp>(data, carry);

		// Store tile
		util::StoreTile<T, SizeT, CtaScan::LOG_LOADS_PER_TILE, CtaScan::LOG_LOAD_VEC_SIZE, CtaScan::THREADS, CtaScan::WRITE_MODIFIER, UNGUARDED_IO>::Invoke(
			data, d_out, cta_offset, out_of_bounds);
	}

	
	/**
	 * Constructor
	 */
	__device__ __forceinline__ CtaScan(
		uint4 smem_pool[ScanKernelConfig::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)],
		T *d_in,
		T *d_out,
		T spine_partial = CtaScan::Identity()) :
			CtaScanBase<typename CtaScan::SrtsGrid>(smem_pool, warpscan),
			carry(spine_partial),			// Seed carry with spine partial
			d_in(d_in),
			d_out(d_out)
	{
		this->template Initialize<CtaScan::Identity>();
	}

};



} // namespace scan
} // namespace b40c

