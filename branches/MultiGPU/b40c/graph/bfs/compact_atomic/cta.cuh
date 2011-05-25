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
 * Tile-processing functionality for BFS compaction upsweep kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/initialize_tile.cuh>
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/io/scatter_tile.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/util/operators.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace compact_atomic {


/**
 * Derivation of KernelPolicy that encapsulates tile-processing routines
 */
template <typename KernelPolicy>
struct Cta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelPolicy::VertexId 		VertexId;
	typedef typename KernelPolicy::ValidFlag		ValidFlag;
	typedef typename KernelPolicy::CollisionMask 	CollisionMask;
	typedef typename KernelPolicy::SizeT 			SizeT;
	typedef typename KernelPolicy::ThreadId			ThreadId;
	typedef typename KernelPolicy::SrtsDetails 		SrtsDetails;
	typedef typename KernelPolicy::SmemStorage		SmemStorage;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS queue index
	VertexId 				queue_index;

	// Input and output device pointers
	VertexId 				*d_in;					// Incoming vertex ids
	VertexId 				*d_out;					// Compacted vertex ids
	VertexId 				*d_parent_in;			// Incoming parent vertex ids (optional)
	VertexId 				*d_parent_out;			// Compacted parent vertex ids (optional)
	CollisionMask 			*d_collision_cache;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for SRTS scan grid
	SrtsDetails 			srts_details;

	// Smem storage for reduction tree and hashing scratch
	volatile VertexId 		(*vid_hashtable)[SmemStorage::WARP_HASH_ELEMENTS];
	volatile VertexId		*history;



	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_ids[LOADS_PER_TILE][LOAD_VEC_SIZE];
		VertexId 	parent_ids[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Whether or not the corresponding vertex_ids is valid for exploring
		ValidFlag 	flags[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Tile of local scatter offsets
		SizeT 		ranks[LOADS_PER_TILE][LOAD_VEC_SIZE];

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * InitFlags
			 */
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				// Initially valid if vertex-id is valid
				tile->flags[LOAD][VEC] = (tile->vertex_ids[LOAD][VEC] == -1) ? 0 : 1;

				// Next
				Iterate<LOAD, VEC + 1>::InitFlags(tile);
			}

			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					// Location of mask byte to read
					SizeT mask_byte_offset = (tile->vertex_ids[LOAD][VEC] & KernelPolicy::VERTEX_ID_MASK) >> 3;

					// Bit in mask byte corresponding to current vertex id
					CollisionMask mask_bit = 1 << (tile->vertex_ids[LOAD][VEC] & 7);

					// Read byte from from collision cache bitmask
					CollisionMask mask_byte;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						mask_byte, cta->d_collision_cache + mask_byte_offset);

					if (mask_bit & mask_byte) {

						// Seen it
						tile->flags[LOAD][VEC] = 0;

					} else {

						// Update with best effort
						mask_byte |= mask_bit;
						util::io::ModifiedStore<util::io::st::cg>::St(
							mask_byte,
							cta->d_collision_cache + mask_byte_offset);
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::BitmaskCull(cta, tile);
			}


			/**
			 * HistoryCull
			 */
			static __device__ __forceinline__ void HistoryCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					int hash = ((typename KernelPolicy::UnsignedBits) tile->vertex_ids[LOAD][VEC]) % SmemStorage::HISTORY_HASH_ELEMENTS;
					VertexId retrieved = cta->history[hash];

					if (retrieved == tile->vertex_ids[LOAD][VEC]) {
						// Seen it
						tile->flags[LOAD][VEC] = 0;

					} else {
						// Update it
						cta->history[hash] = tile->vertex_ids[LOAD][VEC];
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::HistoryCull(cta, tile);
			}


			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(
				Cta *cta,
				Tile *tile)
			{
				if (tile->flags[LOAD][VEC]) {

					int warp_id 		= threadIdx.x >> 5;
					int hash 			= tile->vertex_ids[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

					cta->vid_hashtable[warp_id][hash] = tile->vertex_ids[LOAD][VEC];
					VertexId retrieved = cta->vid_hashtable[warp_id][hash];

					if (retrieved == tile->vertex_ids[LOAD][VEC]) {

						cta->vid_hashtable[warp_id][hash] = threadIdx.x;
						VertexId tid = cta->vid_hashtable[warp_id][hash];
						if (tid != threadIdx.x) {
							tile->flags[LOAD][VEC] = 0;
						}
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::WarpCull(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::InitFlags(tile);
			}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
			}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
			}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::WarpCull(cta, tile);
			}
		};



		/**
		 * Terminate iteration
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// InitFlags
			static __device__ __forceinline__ void InitFlags(Tile *tile) {}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(Cta *cta, Tile *tile) {}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(Cta *cta, Tile *tile) {}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(Cta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Initializer
		 */
		__device__ __forceinline__ void InitFlags()
		{
			Iterate<0, 0>::InitFlags(this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void BitmaskCull(Cta *cta)
		{
			Iterate<0, 0>::BitmaskCull(cta, this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void LocalCull(Cta *cta)
		{
			Iterate<0, 0>::HistoryCull(cta, this);
			Iterate<0, 0>::WarpCull(cta, this);
		}
	};




	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	__device__ __forceinline__ Cta(
		VertexId				queue_index,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		VertexId 				*d_out,
		VertexId 				*d_parent_in,
		VertexId 				*d_parent_out,
		CollisionMask 			*d_collision_cache,
		util::CtaWorkProgress	&work_progress) :

			queue_index(queue_index),
			srts_details(
				smem_storage.state.smem_pool.raking_elements,
				smem_storage.state.warpscan,
				0),
			vid_hashtable(smem_storage.state.smem_pool.vid_hashtable),
			history(smem_storage.history),

			d_in(d_in),
			d_out(d_out),
			d_parent_in(d_parent_in),
			d_parent_out(d_parent_out),
			d_collision_cache(d_collision_cache),
			work_progress(work_progress)

	{
		// Initialize history duplicate-filter
		for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelPolicy::THREADS) {
			history[offset] = -1;
		}
	}


	/**
	 * Process a single, full tile
	 */
	__device__ __forceinline__ void ProcessTile(
		SizeT cta_offset,
		const SizeT &guarded_elements = KernelPolicy::TILE_ELEMENTS)
	{
		Tile<KernelPolicy::LOG_LOADS_PER_TILE, KernelPolicy::LOG_LOAD_VEC_SIZE> tile;

		// Load tile
		util::io::LoadTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE,
			KernelPolicy::THREADS,
			KernelPolicy::READ_MODIFIER>::LoadValid(
				tile.vertex_ids,
				(VertexId) -1,
				d_in + cta_offset,
				guarded_elements);

		// Init valid flags
		tile.InitFlags();

		// Cull using global collision bitmask
		tile.BitmaskCull(this);

		// Cull using local collision hashing
		tile.LocalCull(this);

		// Copy flags into ranks
		util::io::InitializeTile<
			KernelPolicy::LOG_LOADS_PER_TILE,
			KernelPolicy::LOG_LOAD_VEC_SIZE>::Copy(tile.ranks, tile.flags);

		// Protect repurposable storage that backs both raking lanes and local cull scratch
		__syncthreads();

		// Scan tile of ranks, using an atomic add to reserve
		// space in the compacted queue, seeding ranks
		util::scan::CooperativeTileScan<
			SrtsDetails,
			KernelPolicy::LOAD_VEC_SIZE,
			true,							// exclusive
			util::Operators<SizeT>::Sum>::ScanTileWithEnqueue(
				srts_details,
				tile.ranks,
				work_progress.GetQueueCounter<SizeT>(queue_index + 1));

		// Protect repurposable storage that backs both raking lanes and local cull scratch
		__syncthreads();

		// Scatter directly (without first compacting in smem scratch), predicated
		// on flags (treat vertex_ids, flags, and ranks as linear arrays)
		util::io::ScatterTile<
			KernelPolicy::TILE_ELEMENTS_PER_THREAD,
			KernelPolicy::THREADS,
			KernelPolicy::WRITE_MODIFIER>::Scatter(
				d_out,
				(VertexId *) tile.vertex_ids,
				(ValidFlag *) tile.flags,
				(SizeT *) tile.ranks);

		if (KernelPolicy::MARK_PARENTS) {

			// Compact parent vertices as well
			util::io::LoadTile<
				KernelPolicy::LOG_LOADS_PER_TILE,
				KernelPolicy::LOG_LOAD_VEC_SIZE,
				KernelPolicy::THREADS,
				KernelPolicy::READ_MODIFIER>::LoadValid(
					tile.parent_ids,
					d_parent_in + cta_offset,
					guarded_elements);

			// Scatter valid vertex_ids, predicated on flags (treat
			// vertex_ids, flags, and ranks as linear arrays)
			util::io::ScatterTile<
				KernelPolicy::TILE_ELEMENTS_PER_THREAD,
				KernelPolicy::THREADS,
				KernelPolicy::WRITE_MODIFIER>::Scatter(
					d_parent_out,
					(VertexId *) tile.parent_ids,
					(ValidFlag *) tile.flags,
					(SizeT *) tile.ranks);
		}
	}
};


} // namespace compact_atomic
} // namespace bfs
} // namespace graph
} // namespace b40c

