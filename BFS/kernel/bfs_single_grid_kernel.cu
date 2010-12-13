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
 * 
 * A single-grid breadth-first-search kernel.  (BFS-SG)
 * 
 ******************************************************************************/

#pragma once

#include <bfs_kernel_common.cu>

namespace b40c {

/******************************************************************************
 * BFS-SG Granularity Configuration 
 ******************************************************************************/

//  CTA size in threads
#define B40C_BFS_SG_SM20_LOG_CTA_THREADS(strategy)				((strategy == EXPAND_CONTRACT) ? 7 : 7)				// 128 threads on GF100		 
#define B40C_BFS_SG_SM12_LOG_CTA_THREADS(strategy)				((strategy == EXPAND_CONTRACT) ? 8 : 8)		 		// 128 threads on GT200
#define B40C_BFS_SG_SM10_LOG_CTA_THREADS(strategy)				((strategy == EXPAND_CONTRACT) ? 8 : 8)				// 128 threads on G80
#define B40C_BFS_SG_LOG_CTA_THREADS(sm_version, strategy)		((sm_version >= 200) ? B40C_BFS_SG_SM20_LOG_CTA_THREADS(strategy) : 	\
																 (sm_version >= 120) ? B40C_BFS_SG_SM12_LOG_CTA_THREADS(strategy) : 	\
																					   B40C_BFS_SG_SM10_LOG_CTA_THREADS(strategy))		

// Target CTA occupancy.  Params: SM sm_version
#define B40C_BFS_SG_SM20_OCCUPANCY()							(8)				// 8 threadblocks on GF100
#define B40C_BFS_SG_SM12_OCCUPANCY()							(1)				// 1 threadblocks on GT200
#define B40C_BFS_SG_SM10_OCCUPANCY()							(1)				// 1 threadblocks on G80
#define B40C_BFS_SG_OCCUPANCY(sm_version)						((sm_version >= 200) ? B40C_BFS_SG_SM20_OCCUPANCY() : 	\
																 (sm_version >= 120) ? B40C_BFS_SG_SM12_OCCUPANCY() : 	\
																					   B40C_BFS_SG_SM10_OCCUPANCY())		


// Vector size of load. Params: SM sm_version, algorithm				
// (N.B.: currently only vec-1 for EXPAND_CONTRACT, up to vec-4 for CONTRACT_EXPAND)  
#define B40C_BFS_SG_SM20_LOG_LOAD_VEC_SIZE(strategy)			((strategy == EXPAND_CONTRACT) ? 0 : 0)		 
#define B40C_BFS_SG_SM12_LOG_LOAD_VEC_SIZE(strategy)			((strategy == EXPAND_CONTRACT) ? 0 : 1)		 
#define B40C_BFS_SG_SM10_LOG_LOAD_VEC_SIZE(strategy)			((strategy == EXPAND_CONTRACT) ? 0 : 1)		
#define B40C_BFS_SG_LOG_LOAD_VEC_SIZE(sm_version, strategy)		((sm_version >= 200) ? B40C_BFS_SG_SM20_LOG_LOAD_VEC_SIZE(strategy) : 	\
																 (sm_version >= 120) ? B40C_BFS_SG_SM12_LOG_LOAD_VEC_SIZE(strategy) : 	\
																					   B40C_BFS_SG_SM10_LOG_LOAD_VEC_SIZE(strategy))		


// Number of raking threads.  Params: SM sm_version, strategy			
// (N.B: currently supported up to 1 warp)
#define B40C_BFS_SG_SM20_LOG_RAKING_THREADS()					(B40C_LOG_WARP_THREADS + 0)		// 1 raking warps on GF100
#define B40C_BFS_SG_SM12_LOG_RAKING_THREADS()					(B40C_LOG_WARP_THREADS + 0)		// 1 raking warps on GT200
#define B40C_BFS_SG_SM10_LOG_RAKING_THREADS()					(B40C_LOG_WARP_THREADS + 0)		// 1 raking warps on G80
#define B40C_BFS_SG_LOG_RAKING_THREADS(sm_version, strategy)	((sm_version >= 200) ? B40C_BFS_SG_SM20_LOG_RAKING_THREADS() : 	\
																 (sm_version >= 120) ? B40C_BFS_SG_SM12_LOG_RAKING_THREADS() : 	\
																					   B40C_BFS_SG_SM10_LOG_RAKING_THREADS())		

// Size of sractch space (in bytes).  Params: SM sm_version, strategy
#define B40C_BFS_SG_SM20_SCRATCH_SPACE(strategy)				(45 * 1024 / B40C_BFS_SG_SM20_OCCUPANCY()) 
#define B40C_BFS_SG_SM12_SCRATCH_SPACE(strategy)				(15 * 1024 / B40C_BFS_SG_SM12_OCCUPANCY())
#define B40C_BFS_SG_SM10_SCRATCH_SPACE(strategy)				(7  * 1024 / B40C_BFS_SG_SM10_OCCUPANCY())
#define B40C_BFS_SG_SCRATCH_SPACE(sm_version, strategy)			((sm_version >= 200) ? B40C_BFS_SG_SM20_SCRATCH_SPACE(strategy) : 	\
																 (sm_version >= 120) ? B40C_BFS_SG_SM12_SCRATCH_SPACE(strategy) : 	\
																					   B40C_BFS_SG_SM10_SCRATCH_SPACE(strategy))		

// Number of elements per tile.  Params: SM sm_version, strategy
#define B40C_BFS_SG_LOG_TILE_ELEMENTS(sm_version, strategy)		(B40C_BFS_SG_LOG_CTA_THREADS(sm_version, strategy) + B40C_BFS_SG_LOG_LOAD_VEC_SIZE(sm_version, strategy))

// Number of elements per subtile.  Params: strategy  
#define B40C_BFS_SG_LOG_SUBTILE_ELEMENTS(strategy)				((strategy == EXPAND_CONTRACT) ? 5 : 6)		// 64 for CONTRACT_EXPAND, 32 for EXPAND_CONTRACT


/******************************************************************************
 * Kernel routines
 ******************************************************************************/

/**
 * 
 * A single-grid breadth-first-search kernel.  (BFS-SG)
 * 
 * Marks each node with its distance from the given "source" node.  (I.e., 
 * nodes are marked with the iteration at which they were "discovered").
 *     
 * A BFS search iteratively expands outwards from the given source node.  At 
 * each iteration, the algorithm discovers unvisited nodes that are adjacent 
 * to the nodes discovered by the previous iteration.  The first iteration 
 * discovers the source node. 
 * 
 * All iterations are performed by a single kernel-launch.  This is 
 * made possible by software global-barriers across threadblocks.  
 * 
 * The algorithm strategy is either:
 *   (a) Contract-then-expand
 *   (b) Expand-then-contract
 * For more details, see the enum type BfsStrategy
 *   
 *
 */
template <
	typename IndexType, 
	int STRATEGY,			// Should be of type "BfsStrategy": NVBUGS 768132
	bool INSTRUMENTED> 						 
__launch_bounds__ (
	1 << B40C_BFS_SG_LOG_CTA_THREADS(__CUDA_ARCH__, STRATEGY), 
	B40C_BFS_SG_OCCUPANCY(__CUDA_ARCH__))
__global__ void BfsSingleGridKernel(
	IndexType src,										// Source node for the first iteration
	IndexType *d_in_queue,								// Queue of node-IDs to consume
	IndexType *d_out_queue,								// Queue of node-IDs to produce
	IndexType *d_column_indices,						// CSR column indices
	IndexType *d_row_offsets,							// CSR row offsets 
	IndexType *d_source_dist,							// Distance from the source node (initialized to -1) (per-node)
	int *d_queue_lengths,								// Rotating 4-element array of atomic counters indicating sizes of the incoming and outgoing frontier queues
	int *d_sync,										// Array of global synchronization counters, one for each threadblock
	unsigned long long *d_barrier_time)					// Time (in clocks) spent by each threadblock in software global barrier
{
	const int LOG_CTA_THREADS			= B40C_BFS_SG_LOG_CTA_THREADS(__CUDA_ARCH__, STRATEGY);
	const int CTA_THREADS				= 1 << LOG_CTA_THREADS;
	
	const int TILE_ELEMENTS 			= 1 << B40C_BFS_SG_LOG_TILE_ELEMENTS(__CUDA_ARCH__, STRATEGY);

	const int LOG_SUBTILE_ELEMENTS		= B40C_BFS_SG_LOG_SUBTILE_ELEMENTS(STRATEGY);
	const int SUBTILE_ELEMENTS			= 1 << LOG_SUBTILE_ELEMENTS;		
	
	const int SCRATCH_SPACE				= B40C_BFS_SG_SCRATCH_SPACE(__CUDA_ARCH__, STRATEGY) / sizeof(IndexType);
	const int LOAD_VEC_SIZE				= 1 << B40C_BFS_SG_LOG_LOAD_VEC_SIZE(__CUDA_ARCH__, STRATEGY);
	const int RAKING_THREADS			= 1 << B40C_BFS_SG_LOG_RAKING_THREADS(__CUDA_ARCH__, STRATEGY);
	
	// Number of scan partials for a tile
	const int LOG_SCAN_PARTIALS 		= LOG_CTA_THREADS;			// One partial per thread
	const int SCAN_PARTIALS				= 1 << LOG_SCAN_PARTIALS;

	// Number of scan partials per raking segment
	const int LOG_PARTIALS_PER_SEG		= LOG_SCAN_PARTIALS - B40C_LOG_WARP_THREADS;					
	const int PARTIALS_PER_SEG			= 1 << LOG_PARTIALS_PER_SEG;
	
	// Number of scan partials per scratch_pool row
	const int LOG_PARTIALS_PER_ROW		= B40C_MAX(B40C_LOG_MEM_BANKS(__CUDA_ARCH__), LOG_PARTIALS_PER_SEG); 	// Floor of MEM_BANKS partials per row
	const int PARTIALS_PER_ROW			= 1 << LOG_PARTIALS_PER_ROW;
	const int PADDED_PARTIALS_PER_ROW 	= PARTIALS_PER_ROW + 1;
	const int SCAN_ROWS 				= SCAN_PARTIALS / PARTIALS_PER_ROW;		// Number of scratch_pool rows for scan 
	
	// Number of raking segments per scratch_pool row
	const int LOG_SEGS_PER_ROW 			= LOG_PARTIALS_PER_ROW - LOG_PARTIALS_PER_SEG;		
	const int SEGS_PER_ROW				= 1 << LOG_SEGS_PER_ROW;

	// Figure out how big our multipurpose scratch_pool allocation should be (in 128-bit int4s)
	const int SCAN_BYTES				= SCAN_ROWS * PADDED_PARTIALS_PER_ROW * sizeof(int);
	const int SCRATCH_BYTES				= SCRATCH_SPACE * sizeof(IndexType);
	const int SHARED_BYTES 				= B40C_MAX(SCAN_BYTES, SCRATCH_BYTES);
	const int SHARED_INT4S				= (SHARED_BYTES + sizeof(int4) - 1) / sizeof(int4);

	// Cache-modifiers
	const CacheModifier QUEUE_MODIFIER						= CG;
	const CacheModifier COLUMN_INDICES_MODIFIER				= CG;
	const CacheModifier SOURCE_DIST_MODIFIER				= CG;
	const CacheModifier ROW_OFFSETS_MODIFIER				= CG;
	const CacheModifier MISALIGNED_ROW_OFFSETS_MODIFIER		= CA;
	
	SuppressUnusedConstantWarning(CTA_THREADS);
	SuppressUnusedConstantWarning(LOAD_VEC_SIZE);
	SuppressUnusedConstantWarning(PARTIALS_PER_SEG);
	SuppressUnusedConstantWarning(QUEUE_MODIFIER);
	SuppressUnusedConstantWarning(COLUMN_INDICES_MODIFIER);
	SuppressUnusedConstantWarning(SOURCE_DIST_MODIFIER);
	SuppressUnusedConstantWarning(ROW_OFFSETS_MODIFIER);
	SuppressUnusedConstantWarning(MISALIGNED_ROW_OFFSETS_MODIFIER);
	
	
	__shared__ int4 aligned_smem_pool[SHARED_INT4S];	// Smem for: (i) raking prefix sum; and (ii) hashing/compaction scratch space
	__shared__ int warpscan[2][B40C_WARP_THREADS];		// Smem for cappping off the local prefix sum
	__shared__ int s_enqueue_offset;					// Current tile's offset into the output queue for the next iteration 

	int* 		scan_pool = reinterpret_cast<int*>(aligned_smem_pool);				// The smem pool for (i) above
	IndexType* 	scratch_pool = reinterpret_cast<IndexType*>(aligned_smem_pool);		// The smem pool for (ii) above
	
	__shared__ int s_num_incoming_nodes;	// Number of elements in the incoming frontier queue for the current iteration
	__shared__ int s_cta_offset;			// Offset in the incoming frontier queue for this CTA to begin raking its tiles
	__shared__ int s_cta_extra_elements;	// Number of elements in a last, partially-full tile (needing guarded loads)   
	__shared__ int s_cta_out_of_bounds;		// The offset in the incoming frontier for this CTA to stop raking full tiles

	unsigned long long barrier_time = 0;
	unsigned int total_queued;
	IndexType iteration = 0;				// Current BFS iteration

	
	//
	// Initialize structures
	//

	// Raking threads determine their scan_pool segment to sequentially rake for smem reduction/scan
	int *raking_segment;						 
	if (threadIdx.x < RAKING_THREADS) {

		int row = threadIdx.x >> LOG_SEGS_PER_ROW;
		int col = (threadIdx.x & (SEGS_PER_ROW - 1)) << LOG_PARTIALS_PER_SEG;
		raking_segment = scan_pool + (row * PADDED_PARTIALS_PER_ROW) + col; 
	}
	
	// Set identity into first half of warpscan 
	if (threadIdx.x < B40C_WARP_THREADS) {
		warpscan[0][threadIdx.x] = 0;
	}
	
	// Determine which scan_pool cell to place my partial reduction into
	int row = threadIdx.x >> LOG_PARTIALS_PER_ROW; 
	int col = threadIdx.x & (PARTIALS_PER_ROW - 1); 
	int *base_partial = scan_pool + (row * PADDED_PARTIALS_PER_ROW) + col;
	
	// First iteration: process the source node specified in the formal parameters
	if (threadIdx.x == 0) {
	
		total_queued = (STRATEGY == EXPAND_CONTRACT) ? 1 : 0;
		s_num_incoming_nodes = 1;
		s_cta_offset = 0;
		s_cta_out_of_bounds = 0;
		
		if (blockIdx.x == 0) {
			
			// First CTA resets global atomic counter for future outgoing counter
			int future_queue_length_idx = (iteration + 2) & 0x3; 	// Index of the future outgoing queue length (must reset for the next iteration) 
			d_queue_lengths[future_queue_length_idx] = 0;

			// Only the first CTA does any work (so enqueue it, and reset outgoing queue length) 
			s_cta_extra_elements = 1;
			d_in_queue[0] = src;
			d_queue_lengths[1] = 0;
			
			// Expand-contract algorithm requires setting source to already-discovered
			if (STRATEGY == EXPAND_CONTRACT) {
				d_source_dist[src] = 0;
			}
		} else {

			// No work for all other CTAs
			s_cta_extra_elements = 0;
		}
	}
	
	__syncthreads();

	//
	// Loop over BFS frontier queue iterations
	//
	do {

		// Index of the outgoing queue length
		int outgoing_queue_length_idx = (iteration + 1) & 0x3; 				

		// Perform a pass through the incoming frontier queue
		if (iteration & 0x1) {
		
			// Odd iteration: frontier queue streamed out -> in 
			BfsIteration<(BfsStrategy) STRATEGY, IndexType, CTA_THREADS, TILE_ELEMENTS, PARTIALS_PER_SEG, SCRATCH_SPACE, 
					LOAD_VEC_SIZE, QUEUE_MODIFIER, COLUMN_INDICES_MODIFIER, SOURCE_DIST_MODIFIER,
					ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER>(
				iteration, scratch_pool, base_partial, raking_segment, warpscan, 
				d_out_queue, d_in_queue, d_column_indices, d_row_offsets, d_source_dist, d_queue_lengths + outgoing_queue_length_idx,
				s_enqueue_offset, s_cta_offset, s_cta_extra_elements, s_cta_out_of_bounds);

		} else {
			
			// Even iteration: frontier queue streamed in -> out  
			BfsIteration<(BfsStrategy) STRATEGY, IndexType, CTA_THREADS, TILE_ELEMENTS, PARTIALS_PER_SEG, SCRATCH_SPACE, 
					LOAD_VEC_SIZE, QUEUE_MODIFIER, COLUMN_INDICES_MODIFIER, SOURCE_DIST_MODIFIER,
					ROW_OFFSETS_MODIFIER, MISALIGNED_ROW_OFFSETS_MODIFIER>(
				iteration, scratch_pool, base_partial, raking_segment, warpscan, 
				d_in_queue, d_out_queue, d_column_indices, d_row_offsets, d_source_dist, d_queue_lengths + outgoing_queue_length_idx,
				s_enqueue_offset, s_cta_offset, s_cta_extra_elements, s_cta_out_of_bounds);
		}

		//
		// Global software barrier to make queues coherent between BFS iterations
		//
		
		clock_t start;
		if (INSTRUMENTED) {
			// Instrumented timer for global barrier
			if (threadIdx.x == 0) {
				start = clock();
			}
		}
		
		GlobalBarrier(d_sync);

		if (INSTRUMENTED) {
			// Instrumented timer for global barrier
			if (threadIdx.x == 0) {
				clock_t stop = clock();
				if (stop >= start) {
					barrier_time += stop - start;
				} else {
					barrier_time += stop + (((clock_t) -1) - start);
				}
			}
		}		
		
		iteration++;
		
		// Calculate our CTA's work range for the next BFS iteration
		if (threadIdx.x == 0) {

			// First CTA resets global atomic counter for future outgoing counter
			if (blockIdx.x == 0) {
				int future_queue_length_idx = (iteration + 2) & 0x3; 	// Index of the future outgoing queue length (must reset for the next iteration) 
				d_queue_lengths[future_queue_length_idx] = 0;
			}

			// Load the size of the incoming frontier queue
			int num_incoming_nodes;
			int incoming_queue_length_idx = (iteration + 0) & 0x3;	// Index of incoming queue length
			ModifiedLoad<int, CG>::Ld(num_incoming_nodes, d_queue_lengths, incoming_queue_length_idx);
			total_queued += num_incoming_nodes;
	
			//
			// Although work is done in "tile"-sized blocks, work is assigned 
			// across CTAs in smaller "subtile"-sized blocks for better 
			// load-balancing on small problems.  We follow our standard pattern 
			// of spreading the subtiles over p CTAs by giving each CTA a batch of 
			// either k or (k + 1) subtiles.  (And the last CTA must account for 
			// its last subtile likely only being partially-full.)
			//
			
			int total_subtiles 			= (num_incoming_nodes + SUBTILE_ELEMENTS - 1) >> LOG_SUBTILE_ELEMENTS;	// round up
			int subtiles_per_cta 		= total_subtiles / gridDim.x;										// round down for the ks
			int extra_subtiles 			= total_subtiles - (subtiles_per_cta * gridDim.x);					// the +1s 
	
			// Compute number of elements and offset at which to start tile processing
			int cta_elements, cta_offset;
			if (blockIdx.x < extra_subtiles) {
				// The first extra_subtiles-CTAs get k+1 subtiles
				cta_elements = (subtiles_per_cta + 1) << LOG_SUBTILE_ELEMENTS;
				cta_offset = cta_elements * blockIdx.x;
			} else if (blockIdx.x < total_subtiles) {
				// The others get k subtiles
				cta_elements = subtiles_per_cta << LOG_SUBTILE_ELEMENTS;
				cta_offset = (cta_elements * blockIdx.x) + (extra_subtiles << LOG_SUBTILE_ELEMENTS);
			} else {
				// Problem small enough that some CTAs don't even a single subtile
				cta_elements = 0;
				cta_offset = 0;
			}
			
			// Compute (i) TILE aligned limit for tile-processing (oob), 
			// and (ii) how many extra guarded-load elements to process 
			// afterward (always less than a full TILE) 
			int cta_out_of_bounds = cta_offset + cta_elements;
			int cta_extra_elements;
			
			if (cta_out_of_bounds > num_incoming_nodes) {
				// The last CTA rounded its last subtile past the end of the queue
				cta_out_of_bounds -= SUBTILE_ELEMENTS;
				cta_elements -= SUBTILE_ELEMENTS;
				cta_extra_elements = (num_incoming_nodes & (SUBTILE_ELEMENTS - 1)) +		// The delta from the previous SUBTILE alignment and the end of the queue 
					(cta_elements & (TILE_ELEMENTS - 1));									// The delta from the previous TILE alignment 
			} else {
				cta_extra_elements = cta_elements & (TILE_ELEMENTS - 1);				// The delta from the previous TILE alignment
			}

			// Store results for the rest of the CTA
			s_num_incoming_nodes = num_incoming_nodes;
			s_cta_offset = cta_offset;
			s_cta_out_of_bounds = cta_out_of_bounds;
			s_cta_extra_elements = cta_extra_elements;
		}
		
		__syncthreads();
	
	} while (s_num_incoming_nodes > 0);
	
	// All done
	
	if (INSTRUMENTED) {
		// Report statistics
		if (threadIdx.x == 0) {
			
			// Stash the barrier time for our block
			d_barrier_time[blockIdx.x] = barrier_time; 
	
			if (blockIdx.x == 0) {
				
				// Stash the total number of queued items and iterations
				d_queue_lengths[4] = total_queued;
				d_queue_lengths[5] = iteration;
			}
		}
	}
} 


} // b40c namespace


