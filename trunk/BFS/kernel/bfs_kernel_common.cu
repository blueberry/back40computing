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
 * Common types and kernel routines for B40C BFS kernels  
 ******************************************************************************/

#pragma once

#include <b40c_kernel_utils.cu>
#include <b40c_vector_types.cu>

namespace b40c {


/******************************************************************************
 * BFS Algorithm and Granularity Configuration 
 ******************************************************************************/

/**
 * CTA size in threads
 */
#define B40C_BFS_SG_LOG_THREADS								(7)			// 128 threads								
#define B40C_BFS_SG_THREADS									(1 << B40C_BFS_SG_LOG_THREADS)	

/**
 * Enumeration of parallel BFS algorithm strategies
 */
enum BfsStrategy {
	
	/**
	 * Contract-then-Expand
	 * 
	 * At each iteration, the frontier queue is comprised of "unvisited-edges", 
	 * i.e. the concatenation of the adjacency lists belonging to the nodes 
	 * that were discovered from the previous iteration. (These unvisited-edges 
	 * are simply the incident node IDs.)  The algorithm discards the edges 
	 * leading to previously-visited nodes, and then expands the edge-lists of 
	 * the remaining (newly-discovered) nodes into the frontier queue for the 
	 * next iteration. As the frontier is streamed through the SMs for each BFS 
	 * iteration, the kernel operates by:
	 * 
	 *  (1) Streaming in tiles of the frontier queue and contracting the unvisited-
	 *      edges by:
	 *        (i)  Removing incident nodes that were discovered by previous 
	 *             iterations
	 *        (ii) A heuristic for removing duplicate incident nodes††. 
	 *         
	 *      The remaining incident nodes are marked as being discovered at this 
	 *      iteration. 
	 *       
	 *  (2) Expanding the newly discovered nodes into their adjacency lists and
	 *      enqueuing these adjacency lists into the outgoing frontier for 
	 *      processing by the next iteration. 
	 */
	CONTRACT_EXPAND,
	
	/**
	 * Expand-then-Contract 
	 * 
	 * At each iteration, the frontier queue is comprised of "discovered nodes" 
	 * from the previous iteration.  The algorithm expands these nodes into 
	 * their edge-lists.  The edges leading to previously-visited nodes are 
	 * discarded.  Then the remaining (newly-discovered) nodes are enqueued 
	 * into the frontier queue for the next iteration. As the frontier is 
	 * streamed through the SMs for each BFS iteration, the kernel operates by:
	 * 
	 *  (1) Streaming in tiles of the frontier queue and expanding those nodes 
	 *      into their adjacency lists into shared-memory scratch space.
	 *         
	 *  (2) Contracting these "unvisited edges" in shared-memory scratch by:
	 *        (i)  Removing incident nodes that were discovered by previous 
	 *             iterations
	 *        (ii) A heuristic for removing duplicate incident nodes††. 
	 *         
	 *      The remaining incident nodes are marked as being discovered at this 
	 *      iteration, and enqueued into the outgoing frontier for processing by 
	 *      the next iteration.
	 */
	EXPAND_CONTRACT
	
	/**
	 * Footnotes:
	 * 
	 *   †† Frontier duplicates exist when a node is neighbor to multiple nodes 
	 *      discovered by the previous iteration.  Although the operation of the 
	 *      algorithm is correct regardless of the number of times a node is 
	 *      discovered within a given iteration, duplicate-removal can drastically 
	 *      reduce the overall work performed.  When the same node is discovered
	 *      concurrently within a given iteration, its entire adjacency list will 
	 *      be duplicated in the next iteration's frontier.  Duplicate-removal is 
	 *      particularly effective for lattice-like graphs: nodes are often 
	 *      discoverable at a given iteration via multiple indicent edges.  
	 */
};


/******************************************************************************
 * BFS Kernel Subroutines 
 ******************************************************************************/

/**
 * Perform a local prefix sum to rank the specified partial_reductions
 * vector, storing the results in the corresponding local_ranks vector.
 * Also performs an atomic-increment at the d_queue_length address with the 
 * aggregate, storing the previous value in s_enqueue_offset.  Returns the 
 * aggregate.  
 * 
 * Needs a subsequent syncthreads for safety of further scratch_pool usage
 * 
 * Currently only supports RAKING_THREADS = B40C_WARP_THREADS.
 * Currently only supports LOADS_PER_TILE = 1.
 */
template <int LOAD_VEC_SIZE, int ELEMENTS_PER_SEGMENT>
__device__ __forceinline__ 
int LocalScanWithAtomicReservation(
	int *base_partial,
	int *raking_segment,
	int warpscan[2][B40C_WARP_THREADS],
	int partial_reductions[LOAD_VEC_SIZE],
	int local_ranks[LOAD_VEC_SIZE],
	int *d_queue_length,
	int &s_enqueue_offset)
{
	// Reduce in registers, placing the result into our smem cell for raking
	base_partial[0] = SerialReduce<int, LOAD_VEC_SIZE>(partial_reductions);

	__syncthreads();

	// Rake-reduce, warpscan, and rake-scan.
	if (threadIdx.x < B40C_WARP_THREADS) {

		// Serial reduce (rake) in smem
		int raked_reduction = SerialReduce<int, ELEMENTS_PER_SEGMENT>(raking_segment);

		// Warpscan
		int seed = WarpScan<B40C_WARP_THREADS, false>(warpscan, raked_reduction);
		
		// Atomic-increment the global counter with our cycle's allocation
		if (threadIdx.x == 0) {
			s_enqueue_offset = atomicAdd(d_queue_length, warpscan[1][B40C_WARP_THREADS - 1]);
		}
		
		// Serial scan (rake) in smem
		SerialScan<int, ELEMENTS_PER_SEGMENT>(raking_segment, seed);
	}

	__syncthreads();

	SerialScan<int, LOAD_VEC_SIZE>(partial_reductions, local_ranks, base_partial[0]);
	
	return warpscan[1][B40C_WARP_THREADS - 1];
}


/**
 * Loads a single IndexType from the specified offset into node_id
 * if in range, otherwise node_id is assigned -1 instead  
 */
template <typename IndexType, int SCRATCH_SPACE, CacheModifier LIST_MODIFIER>
__device__ __forceinline__
void GuardedSingletonLoad(
	IndexType &node_id,			
	int &hash,
	IndexType *scratch_pool,
	IndexType *node_id_list,
	int load_offset,
	int out_of_bounds)							 
{
	if (load_offset < out_of_bounds) {
		ModifiedLoad<IndexType, LIST_MODIFIER>::Ld(node_id, node_id_list, load_offset);
		hash = node_id % SCRATCH_SPACE;
		scratch_pool[hash] = node_id;
	} else {
		node_id = -1;
		hash = SCRATCH_SPACE - 1;
	}
}


/**
 * Uses vector-loads to read a tile of node-IDs from the node_id_list 
 * reference, optionally conditional on bounds-checking.  Performs a 
 * conservative culling of duplicate node-IDs based upon a linear hashing of 
 * the node-IDs.  The corresponding duplicate flag is set to true for a given 
 * node-ID if it can be verified that some other thread will set its own 
 * duplicate flag false for the same node-ID, false otherwise. 
 * 
 * Needs a subsequent syncthreads for safety of further scratch_pool usage
 */
template <
	typename IndexType, 
	int SCRATCH_SPACE, 
	int LOAD_VEC_SIZE, 
	CacheModifier LIST_MODIFIER,
	bool UNGUARDED_IO>
__device__ __forceinline__
void LoadAndCullDuplicates(
	IndexType node_id[LOAD_VEC_SIZE],		// out param
	bool duplicate[LOAD_VEC_SIZE],			// out param
	IndexType *node_id_list,
	int out_of_bounds,							 
	IndexType *scratch_pool)						 
{
	// Hash offset for each node-ID
	int hash[LOAD_VEC_SIZE];		

	// Initially label everything as a duplicate
	#pragma unroll
	for (int COMPONENT = 0; COMPONENT < LOAD_VEC_SIZE; COMPONENT++) {
		duplicate[COMPONENT] = true;				
	}

	// Load node-IDs
	if (UNGUARDED_IO) {
		
		// Use a built-in, vector-typed alias to load straight into node_id array
		typedef typename VecType<IndexType, LOAD_VEC_SIZE>::Type BuiltinVec; 		

		BuiltinVec *node_id_list_vec = (BuiltinVec *) node_id_list;
		BuiltinVec *built_in_alias = (BuiltinVec *) node_id;
		ModifiedLoad<BuiltinVec, LIST_MODIFIER>::Ld(*built_in_alias, node_id_list_vec, threadIdx.x);

		// Hash the node-IDs into smem scratch
		#pragma unroll
		for (int COMPONENT = 0; COMPONENT < LOAD_VEC_SIZE; COMPONENT++) {
			hash[COMPONENT] = node_id[COMPONENT] % SCRATCH_SPACE;
			scratch_pool[hash[COMPONENT]] = node_id[COMPONENT];
		}
		
	} else {
		
		// N.B.: Wish we could unroll here, but can't use inlined ASM instructions
		// in a pragma-unroll.

		if (LOAD_VEC_SIZE > 0) {
			GuardedSingletonLoad<IndexType, SCRATCH_SPACE, LIST_MODIFIER>(
				node_id[0], hash[0], scratch_pool, node_id_list, (B40C_BFS_SG_THREADS * 0) + threadIdx.x, out_of_bounds);
		}
		if (LOAD_VEC_SIZE > 1) {
			GuardedSingletonLoad<IndexType, SCRATCH_SPACE, LIST_MODIFIER>(
				node_id[1], hash[1], scratch_pool, node_id_list, (B40C_BFS_SG_THREADS * 1) + threadIdx.x, out_of_bounds);
		}
		if (LOAD_VEC_SIZE > 2) {
			GuardedSingletonLoad<IndexType, SCRATCH_SPACE, LIST_MODIFIER>(
				node_id[2], hash[2], scratch_pool, node_id_list, (B40C_BFS_SG_THREADS * 2) + threadIdx.x, out_of_bounds);
		}
		if (LOAD_VEC_SIZE > 3) {
			GuardedSingletonLoad<IndexType, SCRATCH_SPACE, LIST_MODIFIER>(
				node_id[3], hash[3], scratch_pool, node_id_list, (B40C_BFS_SG_THREADS * 3) + threadIdx.x, out_of_bounds);
		}
	}
	
	__syncthreads();
	
	// Retrieve what node-IDs "won" at those locations
	int hashed_node_id[LOAD_VEC_SIZE];	
	
	#pragma unroll
	for (int COMPONENT = 0; COMPONENT < LOAD_VEC_SIZE; COMPONENT++) {

		hashed_node_id[COMPONENT] = scratch_pool[hash[COMPONENT]];
		if (hashed_node_id[COMPONENT] != node_id[COMPONENT]) {

			// A different node beat us to this hash cell; we must assume 
			// that we may not be a duplicate
			duplicate[COMPONENT] = false;
		}
	}
	
	__syncthreads();
	
	// For the winners, hash in thread-IDs to select one of the threads
	#pragma unroll
	for (int COMPONENT = 0; COMPONENT < LOAD_VEC_SIZE; COMPONENT++) {
		if (hashed_node_id[COMPONENT] == node_id[COMPONENT]) {
			scratch_pool[hash[COMPONENT]] = threadIdx.x;
		}
	}
	
	__syncthreads();
	
	// See if our thread won out amongst everyone with similar node-IDs 
	#pragma unroll
	for (int COMPONENT = 0; COMPONENT < LOAD_VEC_SIZE; COMPONENT++) {
		if (hashed_node_id[COMPONENT] == node_id[COMPONENT]) {
			if (scratch_pool[hash[COMPONENT]] == threadIdx.x) {

				// We are an authoritative (non-duplicate) thread for this node-ID
				duplicate[COMPONENT] = false;
			}
		}
	}
}
	

/**
 * Inspects an incident node-ID to see if it's been visited already.  If not,
 * we mark its discovery in d_source_dist at this iteration, returning 
 * the length and offset of its neighbor row.  If not, we return zero as the 
 * length of its neighbor row.
 */
template <
	typename IndexType, 
	int SCRATCH_SPACE, 
	CacheModifier SOURCE_DIST_MODIFIER,
	CacheModifier ROW_OFFSETS_MODIFIER,
	CacheModifier MISALIGNED_ROW_OFFSETS_MODIFIER>
__device__ __forceinline__
void InspectAndUpdate(
	IndexType node_id,		
	int &row_offset,				// out param
	int &row_length,				// out param
	IndexType *d_source_dist,
	IndexType *d_row_offsets,
	IndexType iteration)
{
	// Load source distance of node
	int source_dist;
	ModifiedLoad<int, SOURCE_DIST_MODIFIER>::Ld(source_dist, d_source_dist, node_id);

	if (source_dist == -1) {
		// Node is previously unvisited.  Load neighbor row range from d_row_offsets
		int2 row_range;
		if (node_id & 1) {
			// Misaligned
			ModifiedLoad<int, MISALIGNED_ROW_OFFSETS_MODIFIER>::Ld(row_range.x, d_row_offsets, node_id);
			ModifiedLoad<int, MISALIGNED_ROW_OFFSETS_MODIFIER>::Ld(row_range.y, d_row_offsets, node_id + 1);
		} else {
			// Aligned
			int2* d_row_offsets_v2 = reinterpret_cast<int2*>(d_row_offsets + node_id);
			ModifiedLoad<int2, ROW_OFFSETS_MODIFIER>::Ld(row_range, d_row_offsets_v2, 0);
		}
		// Compute row offset and length
		row_offset = row_range.x;
		row_length = row_range.y - row_range.x;

		// Update distance with current iteration
		d_source_dist[node_id] = iteration;
	}
}


/**
 * Process a single tile of work from the current incoming frontier queue
 */
template <
	typename IndexType,
	int PARTIALS_PER_SEG, 
	int SCRATCH_SPACE, 
	int LOAD_VEC_SIZE,
	CacheModifier QUEUE_MODIFIER,
	CacheModifier COLUMN_INDICES_MODIFIER,
	CacheModifier SOURCE_DIST_MODIFIER,
	CacheModifier ROW_OFFSETS_MODIFIER,
	CacheModifier MISALIGNED_ROW_OFFSETS_MODIFIER,
	bool UNGUARDED_IO>
__device__ __forceinline__ 
void BfsTile(
	IndexType iteration,
	IndexType *scratch_pool,
	int *base_partial,
	int *raking_segment,
	int warpscan[2][B40C_WARP_THREADS],
	IndexType *d_in_queue, 
	IndexType *d_out_queue,
	IndexType *d_column_indices,
	IndexType *d_row_offsets,
	IndexType *d_source_dist,
	int *d_queue_length,
	int &s_enqueue_offset,
	int cta_out_of_bounds)
{
	IndexType dequeued_node_id[LOAD_VEC_SIZE];	// Incoming node-IDs to process for this tile
	bool duplicate[LOAD_VEC_SIZE];				// Whether or not the node-ID is a guaranteed duplicate
	IndexType row_offset[LOAD_VEC_SIZE];		// The offset into column_indices for retrieving the neighbor list
	IndexType row_length[LOAD_VEC_SIZE];		// Number of adjacent neighbors
	int local_rank[LOAD_VEC_SIZE];				// Prefix sum of row-lengths, i.e., local rank for where to plop down neighbor list into scratch 
	int row_progress[LOAD_VEC_SIZE];			// Iterator for the neighbor list
	int cta_progress = 0;						// Progress of the CTA as a whole towards writing out all neighbors to the outgoing queue

	// Initialize neighbor-row-length (and progress through that row) to zero.
	#pragma unroll
	for (int COMPONENT = 0; COMPONENT < LOAD_VEC_SIZE; COMPONENT++) {
		row_length[COMPONENT] = 0;
		row_progress[COMPONENT] = 0;
	}
	
	//
	// Dequeue a tile of incident node-IDs to explore and use a heuristic for 
	// culling duplicates
	//

	LoadAndCullDuplicates<IndexType, SCRATCH_SPACE, LOAD_VEC_SIZE, QUEUE_MODIFIER, UNGUARDED_IO>(
		dequeued_node_id,			// out param
		duplicate,					// out param
		d_in_queue,
		cta_out_of_bounds,							 
		scratch_pool);	
	
	__syncthreads();

	//
	// Inspect visitation status of incident node-IDs, acquiring row offsets 
	// and lengths for previously-undiscovered node-IDs
	//
	// N.B.: Wish we could unroll here, but can't use inlined ASM instructions
	// in a pragma-unroll.
	//

	if (LOAD_VEC_SIZE > 0) {
		if ((!duplicate[0]) && (UNGUARDED_IO || (dequeued_node_id[0] != -1))) {
			InspectAndUpdate<IndexType, SCRATCH_SPACE, SOURCE_DIST_MODIFIER, ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER>(
				dequeued_node_id[0], row_offset[0], row_length[0], d_source_dist, d_row_offsets, iteration);
		}
	}
	if (LOAD_VEC_SIZE > 1) {
		if ((!duplicate[1]) && (UNGUARDED_IO || (dequeued_node_id[1] != -1))) {
			InspectAndUpdate<IndexType, SCRATCH_SPACE, SOURCE_DIST_MODIFIER, ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER>(
				dequeued_node_id[1], row_offset[1], row_length[1], d_source_dist, d_row_offsets, iteration);
		}
	}
	if (LOAD_VEC_SIZE > 2) {
		if ((!duplicate[2]) && (UNGUARDED_IO || (dequeued_node_id[2] != -1))) {
			InspectAndUpdate<IndexType, SCRATCH_SPACE, SOURCE_DIST_MODIFIER, ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER>(
				dequeued_node_id[2], row_offset[2], row_length[2], d_source_dist, d_row_offsets, iteration);
		}
	}
	if (LOAD_VEC_SIZE > 3) {
		if ((!duplicate[3]) && (UNGUARDED_IO || (dequeued_node_id[3] != -1))) {
			InspectAndUpdate<IndexType, SCRATCH_SPACE, SOURCE_DIST_MODIFIER, ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER>(
				dequeued_node_id[3], row_offset[3], row_length[3], d_source_dist, d_row_offsets, iteration);
		}
	}
	

	//
	// Perform local scan of neighbor-counts and reserve a spot for them in 
	// the outgoing queue at s_enqueue_offset
	//

	int enqueue_count = LocalScanWithAtomicReservation<LOAD_VEC_SIZE, PARTIALS_PER_SEG>(
		base_partial, raking_segment, warpscan, row_length, local_rank, d_queue_length, s_enqueue_offset);

	__syncthreads();

	
	//
	// Enqueue the adjacency lists of unvisited node-IDs by repeatedly 
	// constructing a set of gather-offsets in the scratch space, and then 
	// having the entire CTA use them to copy adjacency lists from 
	// column_indices to the outgoing frontier queue.
	//

	while (cta_progress < enqueue_count) {
	
		//
		// Fill the scratch space with gather-offsets for neighbor-lists.  Wish we could 
		// pragma unroll here, but we can't do that with inner loops
		// 

		if (LOAD_VEC_SIZE > 0) {
			const int COMPONENT = 0;
			// Attempt to make futher progress on neighbor list
			int scratch_offset = local_rank[COMPONENT] + row_progress[COMPONENT] - cta_progress;
			while ((row_progress[COMPONENT] < row_length[COMPONENT]) && (scratch_offset < SCRATCH_SPACE)) {
				
				// Put a gather offset into the scratch space
				scratch_pool[scratch_offset] = row_offset[COMPONENT] + row_progress[COMPONENT];
				row_progress[COMPONENT]++;
				scratch_offset++;
			}
		}
		if (LOAD_VEC_SIZE > 1) {
			const int COMPONENT = 1;
			// Attempt to make futher progress on neighbor list
			int scratch_offset = local_rank[COMPONENT] + row_progress[COMPONENT] - cta_progress;
			while ((row_progress[COMPONENT] < row_length[COMPONENT]) && (scratch_offset < SCRATCH_SPACE)) {
				
				// Put a gather offset into the scratch space
				scratch_pool[scratch_offset] = row_offset[COMPONENT] + row_progress[COMPONENT];
				row_progress[COMPONENT]++;
				scratch_offset++;
			}
		}
		if (LOAD_VEC_SIZE > 2) {
			const int COMPONENT = 2;
			// Attempt to make futher progress on neighbor list
			int scratch_offset = local_rank[COMPONENT] + row_progress[COMPONENT] - cta_progress;
			while ((row_progress[COMPONENT] < row_length[COMPONENT]) && (scratch_offset < SCRATCH_SPACE)) {
				
				// Put a gather offset into the scratch space
				scratch_pool[scratch_offset] = row_offset[COMPONENT] + row_progress[COMPONENT];
				row_progress[COMPONENT]++;
				scratch_offset++;
			}
		}
		if (LOAD_VEC_SIZE > 3) {
			const int COMPONENT = 3;
			// Attempt to make futher progress on neighbor list
			int scratch_offset = local_rank[COMPONENT] + row_progress[COMPONENT] - cta_progress;
			while ((row_progress[COMPONENT] < row_length[COMPONENT]) && (scratch_offset < SCRATCH_SPACE)) {
				
				// Put a gather offset into the scratch space
				scratch_pool[scratch_offset] = row_offset[COMPONENT] + row_progress[COMPONENT];
				row_progress[COMPONENT]++;
				scratch_offset++;
			}
		}
		
		__syncthreads();
		
		//
		// Copy adjacency lists from column-indices to outgoing queue
		//

		int remainder = B40C_MIN(SCRATCH_SPACE, enqueue_count - cta_progress);
		for (int scratch_offset = threadIdx.x; scratch_offset < remainder; scratch_offset += B40C_BFS_SG_THREADS) {

			// Gather
			int node_id;
			ModifiedLoad<IndexType, COLUMN_INDICES_MODIFIER>::Ld(
				node_id, d_column_indices, scratch_pool[scratch_offset]);
			
			// Scatter
			d_out_queue[s_enqueue_offset + cta_progress + scratch_offset] = node_id;
		}

		cta_progress += SCRATCH_SPACE;
		
		__syncthreads();
	}
}


/**
 * Processes a BFS iteration through the current incoming frontier queue
 */
template <
	BfsStrategy STRATEGY,
	typename IndexType,
	int TILE_ELEMENTS,
	int PARTIALS_PER_SEG, 
	int SCRATCH_SPACE, 
	int LOAD_VEC_SIZE,
	CacheModifier QUEUE_MODIFIER,
	CacheModifier COLUMN_INDICES_MODIFIER,
	CacheModifier SOURCE_DIST_MODIFIER,
	CacheModifier ROW_OFFSETS_MODIFIER,
	CacheModifier MISALIGNED_ROW_OFFSETS_MODIFIER>
__device__ __forceinline__ 
void BfsIteration(
	IndexType iteration,
	IndexType *scratch_pool,
	int *base_partial,
	int *raking_segment,
	int warpscan[2][B40C_WARP_THREADS],
	IndexType *d_in_queue, 
	IndexType *d_out_queue,
	IndexType *d_column_indices,
	IndexType *d_row_offsets,
	IndexType *d_source_dist,
	int *d_queue_length,
	int &s_enqueue_offset,
	int cta_offset, 
	int cta_extra_elements,
	int cta_out_of_bounds)
{
	// Process all of our full-sized tiles (unguarded loads)
	while (cta_offset <= cta_out_of_bounds - TILE_ELEMENTS) {

		BfsTile<IndexType, PARTIALS_PER_SEG, SCRATCH_SPACE, LOAD_VEC_SIZE, 
				QUEUE_MODIFIER, COLUMN_INDICES_MODIFIER, SOURCE_DIST_MODIFIER, 
				ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER, true>( 
			iteration,
			scratch_pool,
			base_partial,
			raking_segment,
			warpscan,
			d_in_queue + cta_offset, 
			d_out_queue,
			d_column_indices,
			d_row_offsets,
			d_source_dist,
			d_queue_length,
			s_enqueue_offset,
			TILE_ELEMENTS);

		cta_offset += TILE_ELEMENTS;
	}

	// Cleanup any remainder elements (guarded_loads)
	if (cta_extra_elements) {
		
		BfsTile<IndexType, PARTIALS_PER_SEG, SCRATCH_SPACE, LOAD_VEC_SIZE, 
				QUEUE_MODIFIER, COLUMN_INDICES_MODIFIER, SOURCE_DIST_MODIFIER, 
				ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER, false>( 
			iteration,
			scratch_pool,
			base_partial,
			raking_segment,
			warpscan,
			d_in_queue + cta_offset, 
			d_out_queue,
			d_column_indices,
			d_row_offsets,
			d_source_dist,
			d_queue_length,
			s_enqueue_offset,
			cta_extra_elements); 
	}
}


} // b40c namespace


