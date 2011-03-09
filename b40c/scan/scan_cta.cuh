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
 * Tile-processing functionality for scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/reduction/reduction_utils.cuh>
#include <b40c/scan/scan_utils.cuh>
#include <b40c/scan/collective_scan.cuh>

namespace b40c {
namespace scan {


/**
 * Derivation of KernelConfig that encapsulates tile-processing
 * routines
 */
template <typename KernelConfig>
struct ScanCta :
	KernelConfig,
	CollectiveScan<typename KernelConfig::SrtsGrid>
{
	typedef typename KernelConfig::T T;
	typedef typename KernelConfig::SizeT SizeT;

	// The value we will accumulate (in raking threads only)
	T carry;

	// Input and output device pointers
	T *d_in;
	T *d_out;

	// Tile of scan elements
	T data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];

	/**
	 * Process a single tile
	 */
	template <bool UNGUARDED_IO>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{

		// Load tile
		util::LoadTile<T, SizeT, ScanCta::LOG_LOADS_PER_TILE, ScanCta::LOG_LOAD_VEC_SIZE, ScanCta::THREADS, ScanCta::READ_MODIFIER, UNGUARDED_IO>::Invoke(
			data, d_in, cta_offset, out_of_bounds);

		// Scan tile
		this->template ScanTileWithCarry<ScanCta::LOAD_VEC_SIZE, ScanCta::EXCLUSIVE, ScanCta::BinaryOp>(data, carry);

		// Store tile
		util::StoreTile<T, SizeT, ScanCta::LOG_LOADS_PER_TILE, ScanCta::LOG_LOAD_VEC_SIZE, ScanCta::THREADS, ScanCta::WRITE_MODIFIER, UNGUARDED_IO>::Invoke(
			data, d_out, cta_offset, out_of_bounds);
	}

	
	/**
	 * Constructor
	 */
	__device__ __forceinline__ ScanCta(
		uint4 smem_pool[KernelConfig::SMEM_QUADS],
		T warpscan[][B40C_WARP_THREADS(__B40C_CUDA_ARCH__)],
		T *d_in,
		T *d_out,
		T spine_partial = ScanCta::Identity()) :
			CollectiveScan<typename ScanCta::SrtsGrid>(smem_pool, warpscan),
			carry(spine_partial),			// Seed carry with spine partial
			d_in(d_in),
			d_out(d_out)
	{
		this->template Initialize<ScanCta::Identity>();
	}

};



} // namespace scan
} // namespace b40c

