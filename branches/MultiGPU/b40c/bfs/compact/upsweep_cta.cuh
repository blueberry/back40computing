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
	typedef unsigned int 							ThreadId;

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
	VertexId 		*s_vid_hashtable;
	ThreadId		*s_tid_hashtable;
	SizeT 			*s_reduction_tree;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <int LOADS_PER_TILE, int LOAD_VEC_SIZE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			ELEMENTS_PER_TILE = LOADS_PER_TILE * LOAD_VEC_SIZE
		};


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[ELEMENTS_PER_TILE];

		// Whether or not the corresponding vertex_id is valid for exploring
		ValidFlag 	valid[ELEMENTS_PER_TILE];

		// Temporary state for local culling
		int 		hash[ELEMENTS_PER_TILE];			// Hash ids for vertex ids
		bool 		duplicate[ELEMENTS_PER_TILE];		// Status as potential duplicate


		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate over vertex id
		 */
		template <int COUNT, int TOTAL>
		struct Iterate
		{
			/**
			 * BitmaskCull
			 */
			static __device__ __forceinline__ void BitmaskCull(
				UpsweepCta *cta,
				Tile *tile)
			{
				if (tile->valid[COUNT]) {

					// Location of mask byte to read
					SizeT mask_byte_offset = tile->vertex_id[COUNT] >> 3;

					// Bit in mask byte corresponding to current vertex id
					CollisionMask mask_bit = 1 << (tile->vertex_id[COUNT] & 7);

					// Read byte from from collision cache bitmask
					CollisionMask mask_byte;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						mask_byte, cta->d_collision_cache + mask_byte_offset);

					if (mask_bit & mask_byte) {

						// Seen it
						tile->valid[COUNT] = 0;

					} else {

						// Update with best effort
						mask_byte |= mask_bit;
						util::io::ModifiedStore<util::io::st::cg>::St(
							mask_byte,
							cta->d_collision_cache + mask_byte_offset);
					}
				}

				// Next
				Iterate<COUNT + 1, TOTAL>::BitmaskCull(cta, tile);
			}


			/**
			 * HashInVertex
			 */
			static __device__ __forceinline__ void HashInVertex(
				UpsweepCta *cta,
				Tile *tile)
			{
				tile->hash[COUNT] = tile->vertex_id[COUNT] % SmemStorage::SMEM_POOL_VERTEX_IDS;
				tile->duplicate[COUNT] = false;

				// Hash the node-IDs into smem scratch
				if (tile->valid[COUNT]) {
					cta->s_vid_hashtable[tile->hash[COUNT]] = tile->vertex_id[COUNT];
				}

				// Next
				Iterate<COUNT + 1, TOTAL>::HashInVertex(cta, tile);
			}


			/**
			 * HashOutVertex
			 */
			static __device__ __forceinline__ void HashOutVertex(
				UpsweepCta *cta,
				Tile *tile)
			{
				// Retrieve what vertices "won" at the hash locations. If a
				// different node beat us to this hash cell; we must assume
				// that we may not be a duplicate.  Otherwise assume that
				// we are a duplicate... for now.

				if (tile->valid[COUNT]) {
					VertexId hashed_node = cta->s_vid_hashtable[tile->hash[COUNT]];
					tile->duplicate[COUNT] = (hashed_node == tile->vertex_id[COUNT]);
				}

				// Next
				Iterate<COUNT + 1, TOTAL>::HashOutVertex(cta, tile);
			}


			/**
			 * HashInTid
			 */
			static __device__ __forceinline__ void HashInTid(
				UpsweepCta *cta,
				Tile *tile)
			{
				// For the possible-duplicates, hash in thread-IDs to select
				// one of the threads to be the unique one
				if (tile->duplicate[COUNT]) {
					cta->s_tid_hashtable[tile->hash[COUNT]] = threadIdx.x;
				}

				// Next
				Iterate<COUNT + 1, TOTAL>::HashInTid(cta, tile);
			}


			/**
			 * HashOutTid
			 */
			static __device__ __forceinline__ void HashOutTid(
				UpsweepCta *cta,
				Tile *tile)
			{
				// See if our thread won out amongst everyone with similar node-IDs
				if (tile->duplicate[COUNT]) {
					// If not equal to our tid, we are not an authoritative thread
					// for this node-ID
					if (cta->s_tid_hashtable[tile->hash[COUNT]] != threadIdx.x) {
						tile->valid[COUNT] = 0;
					}
				}

				// Next
				Iterate<COUNT + 1, TOTAL>::HashOutTid(cta, tile);
			}


			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->valid[COUNT] = 1;

				// Next
				Iterate<COUNT + 1, TOTAL>::Init(tile);
			}
		};


		/**
		 * Terminate iteration
		 */
		template <int TOTAL>
		struct Iterate<TOTAL, TOTAL>
		{
			// BitmaskCull
			static __device__ __forceinline__ void BitmaskCull(UpsweepCta *cta, Tile *tile) {}

			// HashInVertex
			static __device__ __forceinline__ void HashInVertex(UpsweepCta *cta, Tile *tile) {}

			// HashOutVertex
			static __device__ __forceinline__ void HashOutVertex(UpsweepCta *cta, Tile *tile) {}

			// HashInTid
			static __device__ __forceinline__ void HashInTid(UpsweepCta *cta, Tile *tile) {}

			// HashOutTid
			static __device__ __forceinline__ void HashOutTid(UpsweepCta *cta, Tile *tile) {}

			// Init
			static __device__ __forceinline__ void Init(Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		__device__ __forceinline__ Tile()
		{
			Iterate<0, ELEMENTS_PER_TILE>::Init(this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void BitmaskCull(UpsweepCta *cta)
		{
			Iterate<0, ELEMENTS_PER_TILE>::BitmaskCull(cta, this);
		}

		/**
		 * Culls vertices based upon whether or not we've set a bit for them
		 * in the d_collision_cache bitmask
		 */
		__device__ __forceinline__ void LocalCull(UpsweepCta *cta)
		{
			// Hash the node-IDs into smem scratch
			Iterate<0, ELEMENTS_PER_TILE>::HashInVertex(cta, this);

			__syncthreads();

			// Retrieve what node-IDs "won" at those locations
			Iterate<0, ELEMENTS_PER_TILE>::HashOutVertex(cta, this);

			__syncthreads();

			// For the winners, hash in thread-IDs to select one of the threads
			Iterate<0, ELEMENTS_PER_TILE>::HashInTid(cta, this);

			__syncthreads();

			// See if our thread won out amongst everyone with similar node-IDs
			Iterate<0, ELEMENTS_PER_TILE>::HashOutTid(cta, this);

			__syncthreads();
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
			s_vid_hashtable((VertexId *) smem_storage.smem_pool_int4s),
			s_tid_hashtable((ThreadId *) smem_storage.smem_pool_int4s),
			s_reduction_tree((SizeT *) smem_storage.smem_pool_int4s),
			d_in(d_in),
			d_flags_out(d_flags_out),
			d_spine(d_spine),
			d_collision_cache(d_collision_cache) {}


	/**
	 * Process a single, full tile
	 *
	 * Each thread reduces only the strided values it loads.
	 */
	__device__ __forceinline__ void ProcessFullTile(
		SizeT cta_offset)
	{
		Tile<KernelConfig::LOADS_PER_TILE, KernelConfig::LOAD_VEC_SIZE> tile;

		// Load full tile
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::READ_MODIFIER,
			true>::Invoke(
				reinterpret_cast<VertexId (*)[KernelConfig::LOAD_VEC_SIZE]>(tile.vertex_id),
				d_in,
				cta_offset);

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
			true>::Invoke(
				reinterpret_cast<ValidFlag (*)[KernelConfig::LOAD_VEC_SIZE]>(tile.valid),
				d_flags_out,
				cta_offset);

		// Reduce into carry
		carry += util::reduction::SerialReduce<
			ValidFlag,
			KernelConfig::TILE_ELEMENTS_PER_THREAD>::Invoke(tile.valid);
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
			Tile <1, 1> tile;
			util::io::ModifiedLoad<KernelConfig::READ_MODIFIER>::Ld(
				tile.vertex_id[0],
				d_in + cta_offset);
/*
			printf("Upsweep block %d thread %d looking at vertex %d @ %llu\n",
				blockIdx.x, threadIdx.x, tile.vertex_id[0], (unsigned long long) (d_in + cta_offset));
*/
			// Cull valid flags using global collision bitmask
			tile.BitmaskCull(this);

			// Store flags
			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
				tile.valid[0],
				d_flags_out + cta_offset);

			// Reduce into carry
			carry += tile.valid[0];

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
		carry = util::reduction::TreeReduce<
			SizeT,
			KernelConfig::LOG_THREADS,
			util::DefaultSum<SizeT> >::Invoke<false>( 		// No need to return aggregate reduction in all threads
				carry,
				s_reduction_tree);

		// Write output
		if (threadIdx.x == 0) {
/*
			printf("Block %d thread %d reduced %d valid queue items\n",
				blockIdx.x, threadIdx.x, carry);
*/
			util::io::ModifiedStore<KernelConfig::WRITE_MODIFIER>::St(
				carry, d_spine + blockIdx.x);
		}
	}
};


} // namespace compact
} // namespace bfs
} // namespace b40c

