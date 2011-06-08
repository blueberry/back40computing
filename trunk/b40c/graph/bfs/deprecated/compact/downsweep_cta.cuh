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
 * Derivation of KernelPolicy that encapsulates tile-processing routines
 */
template <typename KernelPolicy>
struct DownsweepCta
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::ValidFlag		ValidFlag;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::SrtsDetails 		SrtsDetails;


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
	VertexId 		*exchange;

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
				smem_storage.smem_pool.raking_elements,
				smem_storage.warpscan,
				0),
			exchange(smem_storage.smem_pool.exchange),
			d_in(d_in),
			d_parent_in(d_parent_in),
			d_flags_in(d_flags_in),
			d_out(d_out),
			d_parent_out(d_parent_out),
			carry(spine_partial) {}			// Seed carry with spine partial

	/**
	 * Process a single tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		SizeT guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		VertexId 		vertex_ids[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];	// Tile of vertex ids
		ValidFlag 		flags[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Tile of valid flags
		SizeT 			ranks[KernelPolicy::LOADS_PER_TILE][KernelPolicy::LOAD_VEC_SIZE];		// Tile of local scatter offsets

		// Load tile of vertex ids
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				vertex_ids,
				d_in + cta_offset,
				guarded_elements);

		// Load tile of valid flags
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				flags,
				(ValidFlag) 0,
				d_flags_in + cta_offset,
				guarded_elements);

		// Copy flags into ranks
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(ranks, flags);

		// Scan tile of ranks
		SizeT valid_elements = util::scan::CooperativeTileScan<
			SrtsDetails,
			KernelPolicy::LOAD_VEC_SIZE,
			true,							// exclusive
			util::DefaultSum>::ScanTile(
				srts_details, ranks);


		// Scatter directly (without first compacting in smem scratch), predicated
		// on flags (treat vertex_ids, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				d_out + carry,
				(VertexId *) vertex_ids,
				(ValidFlag *) flags,
				(SizeT *) ranks);

		if (KernelPolicy::MARK_PARENTS) {

			// Compact parent vertices as well
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::LoadValid(
					vertex_ids,
					d_parent_in + cta_offset,
					guarded_elements);

			// Scatter valid vertex_ids into smem exchange, predicated on flags (treat
			// vertex_ids, flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					d_parent_out + carry,
					(VertexId *) vertex_ids,
					(ValidFlag *) flags,
					(SizeT *) ranks);
		}

/*

		//
		// Scatter by first compacting in smem scratch
		//

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Scatter valid vertex_ids into smem exchange, predicated on flags (treat
		// vertex_ids, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			util::io::st::NONE>::Scatter(
				exchange,
				reinterpret_cast<VertexId *>(vertex_ids),
				reinterpret_cast<ValidFlag *>(flags),
				reinterpret_cast<SizeT *>(ranks));

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		// Gather compacted vertex_ids from smem exchange (in 1-element stride loads)
		VertexId compacted_data[KernelPolicy::TILE_ELEMENTS_PER_THREAD][1];
		util::io::LoadTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelPolicy::THREADS,
			util::io::ld::NONE,
			false>::Invoke(								// Guarded loads
				compacted_data, exchange, valid_elements);

		// Scatter compacted vertex_ids to global output
		util::io::StoreTile<
			KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
			0, 											// Vec-1
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER,
			false>::Invoke(								// Guarded stores
				compacted_data, d_out + carry, valid_elements);

		// Barrier sync to protect smem exchange storage
		__syncthreads();

		if (KernelPolicy::MARK_PARENTS) {

			// Compact parent vertices as well
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER,
				FULL_TILE>::Invoke(vertex_ids, d_parent_in + cta_offset, guarded_elements);

			// Scatter valid vertex_ids into smem exchange, predicated on flags (treat
			// vertex_ids, flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				util::io::st::NONE>::Scatter(
					exchange,
					reinterpret_cast<VertexId *>(vertex_ids),
					reinterpret_cast<ValidFlag *>(flags),
					reinterpret_cast<SizeT *>(ranks));

			// Barrier sync to protect smem exchange storage
			__syncthreads();

			// Gather compacted vertex_ids from smem exchange (in 1-element stride loads)
			VertexId compacted_data[KernelPolicy::TILE_ELEMENTS_PER_THREAD][1];
			util::io::LoadTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0, 											// Vec-1
				KernelPolicy::THREADS,
				util::io::ld::NONE,
				false>::Invoke(								// Guarded loads
					compacted_data, exchange, valid_elements);

			// Scatter compacted vertex_ids to global output
			util::io::StoreTile<
				KernelPolicy::LOG_TILE_ELEMENTS_PER_THREAD,
				0, 											// Vec-1
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER,
				false>::Invoke(								// Guarded stores
					compacted_data, d_parent_out + carry, valid_elements);

			// Barrier sync to protect smem exchange storage
			__syncthreads();
		}
*/

		// Update running discontinuity count for CTA
		carry += valid_elements;
	}
};



} // namespace compact
} // namespace bfs
} // namespace b40c

