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
 * Tile-processing functionality for BFS expansion kernels
 ******************************************************************************/

#pragma once

#include <b40c/util/device_intrinsics.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/scan/cooperative_scan.cuh>
#include <b40c/util/io/modified_load.cuh>
#include <b40c/util/io/modified_store.cuh>
#include <b40c/util/io/load_tile.cuh>


namespace b40c {
namespace bfs {
namespace expand_atomic {


/**
 * Derivation of KernelConfig that encapsulates tile-processing routines
 */
template <typename KernelConfig, typename SmemStorage>
struct SweepCta : KernelConfig
{
	//---------------------------------------------------------------------
	// Typedefs
	//---------------------------------------------------------------------

	// Row-length cutoff below which we expand neighbors by writing gather
	// offsets into scratch space (instead of gang-pressing warps or the entire CTA)
	static const int SCAN_EXPAND_CUTOFF = B40C_WARP_THREADS(KernelConfig::CUDA_ARCH);

	typedef typename KernelConfig::VertexId 		VertexId;
	typedef typename KernelConfig::SizeT 			SizeT;
	typedef typename KernelConfig::SrtsDetails 		SrtsDetails;
	typedef typename SmemStorage::WarpComm			WarpComm;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Current BFS iteration
	VertexId 				iteration;

	// Input and output device pointers
	VertexId 				*d_in;
	VertexId 				*d_parent_in;
	VertexId 				*d_out;
	VertexId 				*d_parent_out;
	VertexId				*d_column_indices;
	SizeT					*d_row_offsets;
	VertexId				*d_source_path;

	// Work progress
	util::CtaWorkProgress	&work_progress;

	// Operational details for SRTS scan grid
	SrtsDetails 			srts_details;

	// Shared memory channels for intra-warp communication
	volatile WarpComm		&warp_comm;

	// Enqueue offset for neighbors of the current tile
	SizeT					&enqueue_offset;

	// Scratch pools for expanding and sharing neighbor gather offsets and parent vertices
	SizeT					*offset_scratch_pool;
	VertexId				*parent_scratch_pool;


	//---------------------------------------------------------------------
	// Helper Structures
	//---------------------------------------------------------------------

	/**
	 * Tile
	 */
	template <
		int LOG_LOADS_PER_TILE,
		int LOG_LOAD_VEC_SIZE,
		bool FULL_TILE>
	struct Tile
	{
		//---------------------------------------------------------------------
		// Typedefs and Constants
		//---------------------------------------------------------------------

		enum {
			LOADS_PER_TILE 		= 1 << LOG_LOADS_PER_TILE,
			LOAD_VEC_SIZE 		= 1 << LOG_LOAD_VEC_SIZE
		};

		typedef typename util::VecType<SizeT, 2>::Type Vec2SizeT;


		//---------------------------------------------------------------------
		// Members
		//---------------------------------------------------------------------

		// Dequeued vertex ids
		VertexId 	vertex_id[LOADS_PER_TILE][LOAD_VEC_SIZE];
		VertexId 	parent_id[LOADS_PER_TILE][LOAD_VEC_SIZE];

		// Edge list details
		SizeT		row_offset[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_length[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		coarse_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		fine_row_rank[LOADS_PER_TILE][LOAD_VEC_SIZE];
		SizeT		row_progress[LOADS_PER_TILE][LOAD_VEC_SIZE];

		SizeT 		enqueue_count;
		SizeT		progress;

		//---------------------------------------------------------------------
		// Helper Structures
		//---------------------------------------------------------------------

		/**
		 * Iterate next vector element
		 */
		template <int LOAD, int VEC, int dummy = 0>
		struct Iterate
		{
			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				tile->row_length[LOAD][VEC] = 0;
				tile->row_progress[LOAD][VEC] = 0;

				Iterate<LOAD, VEC + 1>::Init(tile);
			}

			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(SweepCta *cta, Tile *tile)
			{
				if (FULL_TILE || (tile->vertex_id[LOAD][VEC] != -1)) {

					// Load source path of node
					VertexId source_path;
					util::io::ModifiedLoad<util::io::ld::cg>::Ld(
						source_path,
						cta->d_source_path + tile->vertex_id[LOAD][VEC]);

					// Load neighbor row range from d_row_offsets
					Vec2SizeT row_range;
					if (tile->vertex_id[LOAD][VEC] & 1) {

						// Misaligned: load separately
						util::io::ModifiedLoad<KernelConfig::ROW_OFFSET_UNALIGNED_READ_MODIFIER>::Ld(
							row_range.x,
							cta->d_row_offsets + tile->vertex_id[LOAD][VEC]);

						util::io::ModifiedLoad<KernelConfig::ROW_OFFSET_UNALIGNED_READ_MODIFIER>::Ld(
							row_range.y,
							cta->d_row_offsets + tile->vertex_id[LOAD][VEC] + 1);

					} else {
						// Aligned: load together
						util::io::ModifiedLoad<KernelConfig::ROW_OFFSET_ALIGNED_READ_MODIFIER>::Ld(
							row_range,
							reinterpret_cast<Vec2SizeT*>(cta->d_row_offsets + tile->vertex_id[LOAD][VEC]));
					}

					if (source_path == -1) {

						// Node is previously unvisited: compute row offset and length
						tile->row_offset[LOAD][VEC] = row_range.x;
						tile->row_length[LOAD][VEC] = row_range.y - row_range.x;

						if (KernelConfig::MARK_PARENTS) {

							// Update source path with parent vertex
							util::io::ModifiedStore<util::io::st::cg>::St(
								tile->parent_id[LOAD][VEC],
								cta->d_source_path + tile->vertex_id[LOAD][VEC]);
						} else {

							// Update source path with current iteration
							util::io::ModifiedStore<util::io::st::cg>::St(
								cta->iteration,
								cta->d_source_path + tile->vertex_id[LOAD][VEC]);
						}
					}
				}

				tile->fine_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < SCAN_EXPAND_CUTOFF) ?
					tile->row_length[LOAD][VEC] : 0;

				tile->coarse_row_rank[LOAD][VEC] = (tile->row_length[LOAD][VEC] < SCAN_EXPAND_CUTOFF) ?
					0 : tile->row_length[LOAD][VEC];

				Iterate<LOAD, VEC + 1>::Inspect(cta, tile);
			}


			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ void ExpandByCta(SweepCta *cta, Tile *tile)
			{
				// CTA-based expansion/loading
				while (__syncthreads_or(tile->row_length[LOAD][VEC] >= KernelConfig::THREADS)) {

					if (tile->row_length[LOAD][VEC] >= KernelConfig::THREADS) {
						// Vie for control of the CTA
						cta->warp_comm[0][0] = threadIdx.x;
					}

					__syncthreads();

					if (threadIdx.x == cta->warp_comm[0][0]) {
						// Got control of the CTA
						cta->warp_comm[0][0] = tile->row_offset[LOAD][VEC];										// start
						cta->warp_comm[0][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
						cta->warp_comm[0][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
						if (KernelConfig::MARK_PARENTS) {
							cta->warp_comm[0][3] = tile->vertex_id[LOAD][VEC];									// parent
						}

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;
					}

					__syncthreads();

					SizeT coop_offset 	= cta->warp_comm[0][0] + threadIdx.x;
					SizeT coop_rank	 	= cta->warp_comm[0][1] + threadIdx.x;
					SizeT coop_oob 		= cta->warp_comm[0][2];

					VertexId parent_id;
					if (KernelConfig::MARK_PARENTS) {
						parent_id = cta->warp_comm[0][3];
					}

					VertexId neighbor_id;
					while (coop_offset < coop_oob) {

						// Gather
						util::io::ModifiedLoad<KernelConfig::COLUMN_READ_MODIFIER>::Ld(
							neighbor_id, cta->d_column_indices + coop_offset);

						// Scatter neighbor
						util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(
							neighbor_id, cta->d_out + coop_rank);

						if (KernelConfig::MARK_PARENTS) {
							// Scatter parent
							util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(
								parent_id, cta->d_parent_out + coop_rank);
						}

						coop_offset += KernelConfig::THREADS;
						coop_rank += KernelConfig::THREADS;
					}
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by warp
			 */
			static __device__ __forceinline__ void ExpandByWarp(SweepCta *cta, Tile *tile)
			{
				// Warp-based expansion/loading
				int warp_id = threadIdx.x >> B40C_LOG_WARP_THREADS(KernelConfig::CUDA_ARCH);
				int lane_id = util::LaneId();

				while (__any(tile->row_length[LOAD][VEC] >= SCAN_EXPAND_CUTOFF)) {

					if (tile->row_length[LOAD][VEC] >= SCAN_EXPAND_CUTOFF) {
						// Vie for control of the warp
						cta->warp_comm[warp_id][0] = lane_id;
					}

					if (lane_id == cta->warp_comm[warp_id][0]) {

						// Got control of the warp
						cta->warp_comm[warp_id][0] = tile->row_offset[LOAD][VEC];									// start
						cta->warp_comm[warp_id][1] = tile->coarse_row_rank[LOAD][VEC];								// queue rank
						cta->warp_comm[warp_id][2] = tile->row_offset[LOAD][VEC] + tile->row_length[LOAD][VEC];		// oob
						if (KernelConfig::MARK_PARENTS) {
							cta->warp_comm[warp_id][3] = tile->vertex_id[LOAD][VEC];								// parent
						}

						// Unset row length
						tile->row_length[LOAD][VEC] = 0;
					}

					SizeT coop_offset 	= cta->warp_comm[warp_id][0] + lane_id;
					SizeT coop_rank 	= cta->warp_comm[warp_id][1] + lane_id;
					SizeT coop_oob 		= cta->warp_comm[warp_id][2];

					VertexId parent_id;
					if (KernelConfig::MARK_PARENTS) {
						parent_id = cta->warp_comm[warp_id][3];
					}

					VertexId neighbor_id;
					while (coop_offset < coop_oob) {

						// Gather
						util::io::ModifiedLoad<KernelConfig::COLUMN_READ_MODIFIER>::Ld(
							neighbor_id, cta->d_column_indices + coop_offset);

						// Scatter neighbor
						util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(
							neighbor_id, cta->d_out + coop_rank);

						if (KernelConfig::MARK_PARENTS) {
							// Scatter parent
							util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(
								parent_id, cta->d_parent_out + coop_rank);
						}

						coop_offset += B40C_WARP_THREADS(KernelConfig::CUDA_ARCH);
						coop_rank += B40C_WARP_THREADS(KernelConfig::CUDA_ARCH);
					}
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByWarp(cta, tile);
			}


			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(SweepCta *cta, Tile *tile)
			{
				// Attempt to make further progress on this dequeued item's neighbor
				// list if its current offset into local scratch is in range
				SizeT scratch_offset = tile->fine_row_rank[LOAD][VEC] + tile->row_progress[LOAD][VEC] - tile->progress;

				while ((tile->row_progress[LOAD][VEC] < tile->row_length[LOAD][VEC]) &&
					(scratch_offset < SmemStorage::SCRATCH_ELEMENTS))
				{
					// Put gather offset into scratch space
					cta->offset_scratch_pool[scratch_offset] = tile->row_offset[LOAD][VEC] + tile->row_progress[LOAD][VEC];

					if (KernelConfig::MARK_PARENTS) {
						// Put dequeued vertex as the parent into scratch space
						cta->parent_scratch_pool[scratch_offset] = tile->vertex_id[LOAD][VEC];
					}

					tile->row_progress[LOAD][VEC]++;
					scratch_offset++;
				}

				// Next vector element
				Iterate<LOAD, VEC + 1>::ExpandByScan(cta, tile);
			}
		};


		/**
		 * Iterate next load
		 */
		template <int LOAD, int dummy>
		struct Iterate<LOAD, LOAD_VEC_SIZE, dummy>
		{
			/**
			 * Init
			 */
			static __device__ __forceinline__ void Init(Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Init(tile);
			}

			/**
			 * Inspect
			 */
			static __device__ __forceinline__ void Inspect(SweepCta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::Inspect(cta, tile);
			}

			/**
			 * Expand by CTA
			 */
			static __device__ __forceinline__ void ExpandByCta(SweepCta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByCta(cta, tile);
			}

			/**
			 * Expand by warp
			 */
			static __device__ __forceinline__ void ExpandByWarp(SweepCta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByWarp(cta, tile);
			}

			/**
			 * Expand by scan
			 */
			static __device__ __forceinline__ void ExpandByScan(SweepCta *cta, Tile *tile)
			{
				Iterate<LOAD + 1, 0>::ExpandByScan(cta, tile);
			}
		};

		/**
		 * Terminate
		 */
		template <int dummy>
		struct Iterate<LOADS_PER_TILE, 0, dummy>
		{
			// Init
			static __device__ __forceinline__ void Init(Tile *tile) {}

			// Inspect
			static __device__ __forceinline__ void Inspect(SweepCta *cta, Tile *tile) {}

			// ExpandByCta
			static __device__ __forceinline__ void ExpandByCta(SweepCta *cta, Tile *tile) {}

			// ExpandByWarp
			static __device__ __forceinline__ void ExpandByWarp(SweepCta *cta, Tile *tile) {}

			// ExpandByScan
			static __device__ __forceinline__ void ExpandByScan(SweepCta *cta, Tile *tile) {}
		};


		//---------------------------------------------------------------------
		// Interface
		//---------------------------------------------------------------------

		/**
		 * Constructor
		 */
		__device__ __forceinline__ Tile()
		{
			Iterate<0, 0>::Init(this);
		}

		/**
		 * Inspect dequeued vertices, updating source path if necessary and
		 * obtaining edge-list details
		 */
		__device__ __forceinline__ void Inspect(SweepCta *cta)
		{
			Iterate<0, 0>::Inspect(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices at CTA-expansion granularity
		 */
		__device__ __forceinline__ void ExpandByCta(SweepCta *cta)
		{
			Iterate<0, 0>::ExpandByCta(cta, this);
		}

		/**
		 * Expands neighbor lists for valid vertices a warp-expansion granularity
		 */
		__device__ __forceinline__ void ExpandByWarp(SweepCta *cta)
		{
			Iterate<0, 0>::ExpandByWarp(cta, this);
		}

		/**
		 * Expands neighbor lists by local scan rank
		 */
		__device__ __forceinline__ void ExpandByScan(SweepCta *cta)
		{
			Iterate<0, 0>::ExpandByScan(cta, this);
		}
	};


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	template <typename SmemStorage>
	__device__ __forceinline__ SweepCta(
		VertexId 				iteration,
		SmemStorage 			&smem_storage,
		VertexId 				*d_in,
		VertexId 				*d_parent_in,
		VertexId 				*d_out,
		VertexId 				*d_parent_out,
		VertexId 				*d_column_indices,
		SizeT 					*d_row_offsets,
		VertexId 				*d_source_path,
		util::CtaWorkProgress	&work_progress) :

			srts_details(
				smem_storage.smem_pool_int4s,
				smem_storage.warpscan,
				0),
			warp_comm(smem_storage.warp_comm),
			enqueue_offset(smem_storage.enqueue_offset),
			offset_scratch_pool((SizeT *) smem_storage.smem_pool_int4s),
			parent_scratch_pool((VertexId *) (smem_storage.smem_pool_int4s + SmemStorage::OFFSET_QUADS)),
			iteration(iteration),
			d_in(d_in),
			d_parent_in(d_parent_in),
			d_out(d_out),
			d_parent_out(d_parent_out),
			d_column_indices(d_column_indices),
			d_row_offsets(d_row_offsets),
			d_source_path(d_source_path),
			work_progress(work_progress) {}


	/**
	 * Converts out-of-bounds vertex-ids to -1
	 */
	static __device__ __forceinline__ void LoadTransform(
		VertexId &vertex_id,
		bool in_bounds)
	{
		if (!in_bounds) {
			vertex_id = -1;
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
		Tile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			FULL_TILE> tile;

		// Load tile
		util::io::LoadTile<
			KernelConfig::LOG_LOADS_PER_TILE,
			KernelConfig::LOG_LOAD_VEC_SIZE,
			KernelConfig::THREADS,
			KernelConfig::QUEUE_READ_MODIFIER,
			FULL_TILE>::template Invoke<VertexId, LoadTransform>(
				tile.vertex_id,
				d_in,
				cta_offset,
				out_of_bounds);

		// Load tile of parents
		if (KernelConfig::MARK_PARENTS) {

			util::io::LoadTile<
				KernelConfig::LOG_LOADS_PER_TILE,
				KernelConfig::LOG_LOAD_VEC_SIZE,
				KernelConfig::THREADS,
				KernelConfig::QUEUE_READ_MODIFIER,
				FULL_TILE>::Invoke(
					tile.parent_id,
					d_parent_in,
					cta_offset,
					out_of_bounds);
/*
			if (tile.vertex_id[0][0] != -1) {
				printf("\tIteration %d block %d thread %d dequeued node %d with parent %d\n",
					iteration, blockIdx.x, threadIdx.x, tile.vertex_id[0][0], tile.parent_id[0][0]);
			}
*/
		}

		// Inspect dequeued vertices, updating source path and obtaining
		// edge-list details
		tile.Inspect(this);

		// Scan tile of row ranks (lengths) with enqueue reservation,
		// turning them into enqueue offsets
		util::scan::CooperativeTileScan<
			SrtsDetails,
			KernelConfig::LOAD_VEC_SIZE,
			true,							// exclusive
			util::DefaultSum>::ScanTileWithEnqueue(
				srts_details,
				tile.coarse_row_rank,
				work_progress.GetQueueCounter<SizeT>(iteration + 1));

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByCta(this);

		// Enqueue valid edge lists into outgoing queue
		tile.ExpandByWarp(this);

		// Scan tile of row ranks (lengths) with enqueue reservation,
		// turning them into enqueue offsets
		tile.enqueue_count = util::scan::CooperativeTileScan<
			SrtsDetails,
			KernelConfig::LOAD_VEC_SIZE,
			true,							// exclusive
			util::DefaultSum>::ScanTile(
				srts_details,
				tile.fine_row_rank);

		// Reserve allocation in outgoing queue
		if (threadIdx.x == 0) {
			enqueue_offset = work_progress.Enqueue<SizeT>(tile.enqueue_count, iteration + 1);
		}

		//
		// Enqueue the adjacency lists of unvisited node-IDs by repeatedly
		// gathering edges into the scratch space, and then
		// having the entire CTA copy the scratch pool into the outgoing
		// frontier queue.
		//

		tile.progress = 0;
		while (tile.progress < tile.enqueue_count) {

			// Fill the scratch space with gather-offsets for neighbor-lists.
			tile.ExpandByScan(this);

			__syncthreads();

			// Copy scratch space into queue
			int scratch_remainder = B40C_MIN(SmemStorage::SCRATCH_ELEMENTS, tile.enqueue_count - tile.progress);

			for (int scratch_offset = threadIdx.x;
				scratch_offset < scratch_remainder;
				scratch_offset += KernelConfig::THREADS)
			{
				// Gather a neighbor
				VertexId neighbor_id;
				util::io::ModifiedLoad<KernelConfig::COLUMN_READ_MODIFIER>::Ld(
					neighbor_id,
					d_column_indices + offset_scratch_pool[scratch_offset]);

				// Scatter it into queue
				util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(
					neighbor_id,
					d_out + enqueue_offset + tile.progress + scratch_offset);

				if (KernelConfig::MARK_PARENTS) {
					// Scatter parent it into queue
					VertexId parent_id = parent_scratch_pool[scratch_offset];
					util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(
						parent_id,
						d_parent_out + enqueue_offset + tile.progress + scratch_offset);
				}
			}

			tile.progress += SmemStorage::SCRATCH_ELEMENTS;

			__syncthreads();
		}
	}
};



} // namespace expand_atomic
} // namespace bfs
} // namespace b40c

