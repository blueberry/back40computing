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
 * Tile-processing functionality for consecutive-removal downsweep
 * scan kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/operators.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>

namespace b40c {
namespace bfs {
namespace compact {

/**
 * Shared memory structure
 */
template <typename KernelConfig>
struct DownsweepSmemStorage
{
	uint4 smem_pool_int4s[KernelConfig::SMEM_POOL_QUADS];

	typename KernelConfig::SizeT warpscan[2][B40C_WARP_THREADS(KernelConfig::CUDA_ARCH)];
};


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig, typename SmemStorage>
struct DownsweepCta : KernelConfig
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::VertexId 		VertexId;
	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::SrtsDetails 		SrtsDetails;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running accumulator for the number of discontinuities observed by
	// the CTA over its tile-processing lifetime (managed in each raking thread)
	SizeT 			carry;

	// Input and output device pointers
	VertexId 		*d_in;
	VertexId 		*d_out;
	unsigned char	*d_flags_in;

	// Pool of storage for compacting a tile of values
	VertexId 		*smem_pool;

	// Operational details for SRTS scan grid
	SrtsDetails 	srts_details;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------


	/**
	 * Constructor
	 */
	__device__ __forceinline__ DownsweepCta(
		SmemStorage 	&smem_storage,
		VertexId 		*d_in,
		unsigned char	*d_flags_in,
		VertexId 		*d_out,
		SizeT 			spine_partial) :

			srts_details(smem_storage.smem_pool_int4s, smem_storage.warpscan, 0),
			smem_pool((VertexId*) smem_storage.smem_pool_int4s),
			d_in(d_in),
			d_flags_in(d_flags_in),
			d_out(d_out),
			carry(spine_partial) {}			// Seed carry with spine partial


	/**
	 * Converts out-of-bounds valid-flags to 0
	 */
	static __device__ __forceinline__ void LoadTransform(
		unsigned char &flag,
		bool in_bounds)
	{
		if (!in_bounds) {
			flag = 0;
		}
	}


	/**
	 * Process a single tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		VertexId 		data[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];		// Tile of vertex ids
		unsigned char 	flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];		// Tile of valid flags
		SizeT 			ranks[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];		// Tile of local scatter offsets

		// Load tile of vertex ids
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::Invoke(data, d_in, cta_offset, out_of_bounds);

		// Load tile of valid flags
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::template Invoke<unsigned char, LoadTransform>(
				flags, d_flags_in, cta_offset, out_of_bounds);

		// Copy flags into ranks
		util::io::InitializeTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE>::Copy(ranks, flags);

		// Scan tile of ranks
		SizeT valid_elements = util::scan::CooperativeTileScan<
			SrtsDetails,
			KernelConfig::LOAD_VEC_SIZE,
			true,							// exclusive
			util::DefaultSum>::ScanTile(
				srts_details, ranks);

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Scatter valid data into smem exchange, predicated on flags (treat
		// data, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			util::io::st::NONE>::Scatter(
				smem_pool,
				reinterpret_cast<VertexId *>(data),
				reinterpret_cast<unsigned char *>(flags),
				reinterpret_cast<SizeT *>(ranks));

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Gather compacted data from smem exchange (in 1-element stride loads)
		VertexId compacted_data[KernelConfig::TILE_ELEMENTS_PER_THREAD][1];
		util::io::LoadTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelConfig::THREADS,
			util::io::ld::NONE,
			false>::Invoke(								// Guarded loads
				compacted_data, smem_pool, 0, valid_elements);

		// Scatter compacted data to global output
		SizeT scatter_out_of_bounds = carry + valid_elements;
		util::io::StoreTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			false>::Invoke(								// Guarded stores
				compacted_data, d_out, carry, scatter_out_of_bounds);

		// Update running discontinuity count for CTA
		carry = scatter_out_of_bounds;

		// Barrier sync to protect smem exchange storage
		__syncthreads();
	}
};



} // namespace compact
} // namespace bfs
} // namespace b40c

