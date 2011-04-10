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
 * Tile-processing functionality for BFS compaction downsweep kernels
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
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig>
struct DownsweepCta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelConfig::VertexId 		VertexId;
	typedef typename KernelConfig::ValidFlag		ValidFlag;
	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::SrtsDetails 		SrtsDetails;


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Running accumulator for the number of discontinuities observed by
	// the CTA over its tile-processing lifetime (managed in each raking thread)
	SizeT 			carry;

	// Global device storage
	VertexId 		*d_in;			// Incoming vertex ids
	VertexId 		*d_parent_in;	// Incoming parent vertex ids (optional)
	VertexId 		*d_out;			// Compacted vertex ids
	VertexId 		*d_parent_out;	// Compacted parent vertex ids (optional)
	ValidFlag		*d_flags_in;	// Validity flags

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
	template <typename SmemStorage>
	__device__ __forceinline__ DownsweepCta(
		SmemStorage 	&smem_storage,
		VertexId 		*d_in,
		VertexId 		*d_parent_in,
		ValidFlag		*d_flags_in,
		VertexId 		*d_out,
		VertexId 		*d_parent_out,
		SizeT 			spine_partial) :

			srts_details(
				smem_storage.smem_pool_int4s,
				smem_storage.warpscan,
				0),
			smem_pool((VertexId*) smem_storage.smem_pool_int4s),
			d_in(d_in),
			d_parent_in(d_parent_in),
			d_flags_in(d_flags_in),
			d_out(d_out),
			d_parent_out(d_parent_out),
			carry(spine_partial) {}			// Seed carry with spine partial


	/**
	 * Converts out-of-bounds valid-flags to 0
	 */
	static __device__ __forceinline__ void LoadTransform(
		ValidFlag &flag,
		bool in_bounds)
	{
		if (!in_bounds) {
			flag = 0;
		}
	}


	/**
	 * Process a single tile
	 */
	template <bool FULL_TILE>
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT out_of_bounds = 0)
	{
		VertexId 		vertex_id[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];	// Tile of vertex ids
		ValidFlag 		flags[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];		// Tile of valid flags
		SizeT 			ranks[KernelConfig::LOADS_PER_TILE][KernelConfig::LOAD_VEC_SIZE];		// Tile of local scatter offsets

		// Load tile of vertex ids
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::Invoke(vertex_id, d_in, cta_offset, out_of_bounds);

		// Load tile of valid flags
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			FULL_TILE>::template Invoke<ValidFlag, LoadTransform>(
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

		// Scatter valid vertex_id into smem exchange, predicated on flags (treat
		// vertex_id, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER>::Scatter(
				d_out + carry,
				reinterpret_cast<VertexId *>(vertex_id),
				reinterpret_cast<ValidFlag *>(flags),
				reinterpret_cast<SizeT *>(ranks));


/*
		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Scatter valid vertex_id into smem exchange, predicated on flags (treat
		// vertex_id, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelConfig::TILE_ELEMENTS_PER_THREAD,
			KernelConfig::THREADS,
			util::io::st::NONE>::Scatter(
				smem_pool,
				reinterpret_cast<VertexId *>(vertex_id),
				reinterpret_cast<ValidFlag *>(flags),
				reinterpret_cast<SizeT *>(ranks));

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Gather compacted vertex_id from smem exchange (in 1-element stride loads)
		VertexId compacted_data[KernelConfig::TILE_ELEMENTS_PER_THREAD][1];
		util::io::LoadTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelConfig::THREADS,
			util::io::ld::NONE,
			false>::Invoke(								// Guarded loads
				compacted_data, smem_pool, 0, valid_elements);

		// Scatter compacted vertex_id to global output
 */
		SizeT scatter_out_of_bounds = carry + valid_elements;
/*
		util::io::StoreTile<
			KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			false>::Invoke(								// Guarded stores
				compacted_data, d_out, carry, scatter_out_of_bounds);

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		if (KernelConfig::MARK_PARENTS) {

			// Compact parent vertices as well
			util::io::LoadTile<
				KernelConfig::LOG_LOADS_PER_TILE,
				KernelConfig::LOG_LOAD_VEC_SIZE,
				KernelConfig::THREADS,
				KernelConfig::READ_MODIFIER,
				FULL_TILE>::Invoke(vertex_id, d_parent_in, cta_offset, out_of_bounds);

			// Scatter valid vertex_id into smem exchange, predicated on flags (treat
			// vertex_id, flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelConfig::TILE_ELEMENTS_PER_THREAD,
				KernelConfig::THREADS,
				util::io::st::NONE>::Scatter(
					smem_pool,
					reinterpret_cast<VertexId *>(vertex_id),
					reinterpret_cast<ValidFlag *>(flags),
					reinterpret_cast<SizeT *>(ranks));

			// Barrier sync to protect smem exchange storage
			__syncthreads();

			// Gather compacted vertex_id from smem exchange (in 1-element stride loads)
			VertexId compacted_data[KernelConfig::TILE_ELEMENTS_PER_THREAD][1];
			util::io::LoadTile<
				KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
				0, 											// Vec-1
				KernelConfig::THREADS,
				util::io::ld::NONE,
				false>::Invoke(								// Guarded loads
					compacted_data, smem_pool, 0, valid_elements);

			// Scatter compacted vertex_id to global output
			util::io::StoreTile<
				KernelConfig::LOG_TILE_ELEMENTS_PER_THREAD,
				0, 											// Vec-1
				KernelConfig::THREADS,
				KernelConfig::WRITE_MODIFIER,
				false>::Invoke(								// Guarded stores
					compacted_data, d_parent_out, carry, scatter_out_of_bounds);

			// Barrier sync to protect smem exchange storage
			__syncthreads();
		}
*/
		// Update running discontinuity count for CTA
		carry = scatter_out_of_bounds;
	}
};



} // namespace compact
} // namespace bfs
} // namespace b40c

