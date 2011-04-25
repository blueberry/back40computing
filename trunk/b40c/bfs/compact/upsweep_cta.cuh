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
#include <b40c/util/io/load_tile.cuh>
#include <b40c/util/io/store_tile.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/util/operators.cuh>
#include <b40c/util/reduction/serial_reduce.cuh>
#include <b40c/util/reduction/tree_reduce.cuh>

namespace b40c {
namespace bfs {
namespace compact {


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig, typename SmemStorage>
struct UpsweepCta
{
	//---------------------------------------------------------------------
	// Typedefs and Constants
	//---------------------------------------------------------------------

	typedef typename KernelConfig::VertexId 		VertexId;
	typedef typename KernelConfig::ValidFlag		ValidFlag;
	typedef typename KernelConfig::CollisionMask 	CollisionMask;
	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::ThreadId			ThreadId;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// The value we will accumulate (in each thread)
	SizeT 			carry;

	// Input and output device pointers
	VertexId 		*d_in;
	ValidFlag		*d_flags_out;
	SizeT 			*d_spine;
	CollisionMask 	*d_collision_cache;

	// Smem storage for reduction tree and hashing scratch
	volatile VertexId 	(*s_vid_hashtable)[SmemStorage::WARP_HASH_ELEMENTS];
	volatile VertexId	*s_history;
	SizeT 				*s_reduction_tree;


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
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Whether or not the corresponding vertex_id is valid for exploring
		ValidFlag 	valid[LOADS_PER_TILE][LOAD_VEC_SIZE];

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
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->valid[LOAD][VEC] = 1;

				// Next
				Iterate<LOAD, VEC + 1>::Init(tile);
			}

			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				UpsweepCta *cta,
				Tile *tile)
			{
				if (tile->valid[LOAD][VEC]) {

					// Location of mask byte to read
					SizeT mask_byte_offset = tile->vertex_id[LOAD][VEC] >> 3;

					// Bit in mask byte corresponding to current vertex id
					CollisionMask mask_bit = 1 << (tile->vertex_id[LOAD][VEC] & 7);

					// Read byte from from collision cache bitmask
					CollisionMask mask_byte;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						mask_byte, cta->d_collision_cache + mask_byte_offset);

					if (mask_bit & mask_byte) {

						// Seen it
						tile->valid[LOAD][VEC] = 0;

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
				UpsweepCta *cta,
				Tile *tile)
			{
				if (tile->valid[LOAD][VEC]) {

					int hash = tile->vertex_id[LOAD][VEC] % SmemStorage::HISTORY_HASH_ELEMENTS;
					VertexId retrieved = cta->s_history[hash];

					if (retrieved == tile->vertex_id[LOAD][VEC]) {
						// Seen it
						tile->valid[LOAD][VEC] = 0;
					} else {
						// Update it
						cta->s_history[hash] = tile->vertex_id[LOAD][VEC];
					}
				}

				// Next
				Iterate<LOAD, VEC + 1>::HistoryCull(cta, tile);
			}


			/**
			 * WarpCull
			 */
			static __device__ __forceinline__ void WarpCull(
				UpsweepCta *cta,
				Tile *tile)
			{
				if (tile->valid[LOAD][VEC]) {

					int warp_id 		= threadIdx.x >> 5;
					int hash 			= tile->vertex_id[LOAD][VEC] & (SmemStorage::WARP_HASH_ELEMENTS - 1);

					cta->s_vid_hashtable[warp_id][hash] = tile->vertex_id[LOAD][VEC];
					VertexId retrieved = cta->s_vid_hashtable[warp_id][hash];

					if (retrieved == tile->vertex_id[LOAD][VEC]) {

						cta->s_vid_hashtable[warp_id][hash] = threadIdx.x;
						VertexId tid = cta->s_vid_hashtable[warp_id][hash];
						if (tid != threadIdx.x) {
							tile->valid[LOAD][VEC] = 0;
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
			// Init
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Init(tile);
			}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(UpsweepCta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::BitmaskCull(cta, tile);
			}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(UpsweepCta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::HistoryCull(cta, tile);
			}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(UpsweepCta *cta, Tile *tile)
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
			// Init
			static __device__ __forceinline__ void Init(Tile *tile) {}

			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(UpsweepCta *cta, Tile *tile) {}

			// HistoryCull
			static __device__ __forceinline__ void HistoryCull(UpsweepCta *cta, Tile *tile) {}

			// WarpCull
			static __device__ __forceinline__ void WarpCull(UpsweepCta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Initializer
		 */
		__device__ __forceinline__ void Init()
		{
			Iterate<0, 0>::Init(this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void BitmaskCull(UpsweepCta *cta)
		{
			Iterate<0, 0>::BitmaskCull(cta, this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void LocalCull(UpsweepCta *cta)
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
	__device__ __forceinline__ UpsweepCta(
		SmemStorage 	&smem_storage,
		VertexId 		*d_in,
		ValidFlag		*d_flags_out,
		SizeT 			*d_spine,
		CollisionMask 	*d_collision_cache) :

			carry(0),
			s_vid_hashtable(smem_storage.smem_pool.hashtables.vid_hashtable),
			s_history(smem_storage.smem_pool.hashtables.history),
			s_reduction_tree(smem_storage.smem_pool.reduction_tree),
			d_in(d_in),
			d_flags_out(d_flags_out),
			d_spine(d_spine),
			d_collision_cache(d_collision_cache)
	{
		for (int offset = threadIdx.x; offset < SmemStorage::HISTORY_HASH_ELEMENTS; offset += KernelConfig::THREADS) {
			s_history[offset] = -1;
		}
	}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		Tile<KernelConfig::LOG_LOADS_PER_TILE, KernelConfig::LOG_LOAD_VEC_SIZE> tile;
		tile.Init();

		// Load full tile
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::Invoke(
				tile.vertex_id,
				d_in + cta_offset);

		// Cull valid flags using global collision bitmask
		tile.BitmaskCull(this);

		// Cull valid flags using local collision hashing
		tile.LocalCull(this);

		// Store flags
		util::io::StoreTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::WRITE_MODIFIER,
			true>::Invoke(tile.valid, d_flags_out + cta_offset);

		// Reduce into carry
		carry += util::reduction::SerialReduce<KernelConfig::TILE_ELEMENTS_PER_THREAD>::Invoke(
			(ValidFlag *) tile.valid);
	}


	/**
	 * Process a single, partial tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessPartialTile(
		SizeT cta_offset,
		SizeT out_of_bounds)
	{
		cta_offset += threadIdx.x;

		// Process loads singly, do non-cooperative culling
		while (cta_offset < out_of_bounds) {

			// Load single-load, single-vec tile
			Tile <0, 0> tile;
			tile.Init();
			util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
				tile.vertex_id[0][0],
				d_in + cta_offset);

			// Cull valid flags using global collision bitmask
			tile.BitmaskCull(this);

			// Store flags
			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
				tile.valid[0][0],
				d_flags_out + cta_offset);

			// Reduce into carry
			carry += tile.valid[0][0];

			cta_offset += KernelConfig::THREADS;
		}
	}


	/**
	 * Unguarded collective reduction across all threads, stores final reduction
	 * to output.  Used to collectively reduce each thread's aggregate after
	 * striding through the input.
	 *
	 * All threads assumed to have valid carry data.
	 */
	__device__ __forceinline__ void OutputToSpine()
	{
		__syncthreads();

		carry = util::reduction::TreeReduce<
			SizeT,
			KernelConfig::LOG_THREADS,
			util::DefaultSum<SizeT> >::Invoke<false>( 		// No need to return aggregate reduction in all threads
				carry,
				s_reduction_tree);

		// Write output
		if (threadIdx.x == 0) {
			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
				carry, d_spine + blockIdx.x);
		}
	}
};


} // namespace compact
} // namespace bfs
} // namespace b40c

