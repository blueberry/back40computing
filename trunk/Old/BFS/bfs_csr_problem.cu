/******************************************************************************
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
 ******************************************************************************/

/******************************************************************************
 * ProblemStorage management structures for BFS problem data
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>

#include <b40c/bfs/problem_type.cuh>

namespace b40c {
namespace bfs {


/**
 * CSR storage management structure for BFS problems.  
 */
template <typename _VertexId, typename _SizeT, bool MARK_PARENTS>
struct BfsCsrProblem
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	typedef ProblemType<
		_VertexId,				// VertexId
		_SizeT,					// SizeT
		unsigned char,			// CollisionMask
		unsigned char, 			// ValidFlag
		MARK_PARENTS>			// MARK_PARENTS
			ProblemType;

	typedef typename ProblemType::VertexId 			VertexId;
	typedef typename ProblemType::SizeT				SizeT;
	typedef typename ProblemType::CollisionMask 	CollisionMask;
	typedef typename ProblemType::ValidFlag 		ValidFlag;

	static const SizeT		DEFAULT_QUEUE_PADDING_PERCENT = 20;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Size of the graph
	SizeT 			nodes;
	SizeT			edges;

	// Standard CSR device storage arrays
	VertexId 		*d_column_indices;
	SizeT 			*d_row_offsets;
	VertexId 		*d_source_path;				// Can be used for source distance or parent pointer

	// Best-effort (bit) mask for keeping track of which vertices we've seen so far
	CollisionMask 	*d_collision_cache;
	
	// Frontier queues
	VertexId 		*d_expand_queue;
	VertexId 		*d_compact_queue;

	// Optional queues for tracking parent vertices
	VertexId 		*d_expand_parent_queue;
	VertexId 		*d_compact_parent_queue;

	// Vector of valid flags for elements in the frontier queue
	ValidFlag 		*d_keep;

	// Maximum size of the queues
	SizeT 			max_expand_queue_size;
	SizeT 			max_compact_queue_size;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	BfsCsrProblem() :
		nodes(0),
		edges(0),
		d_column_indices(NULL),
		d_row_offsets(NULL),
		d_source_path(NULL),
		d_collision_cache(NULL),
		d_keep(NULL),
		d_expand_queue(NULL),
		d_compact_queue(NULL),
		d_expand_parent_queue(NULL),
		d_compact_parent_queue(NULL),
		max_expand_queue_size(0),
		max_compact_queue_size(0) {}


	/**
     * Destructor
     */
    virtual ~BfsCsrProblem()
    {
		if (d_column_indices) 			cudaFree(d_column_indices);
		if (d_row_offsets) 				cudaFree(d_row_offsets);
		if (d_source_path) 				cudaFree(d_source_path);
		if (d_collision_cache) 			cudaFree(d_collision_cache);
		if (d_keep) 					cudaFree(d_keep);
		if (d_expand_queue) 			cudaFree(d_expand_queue);
		if (d_compact_queue) 			cudaFree(d_compact_queue);
		if (d_expand_parent_queue) 		cudaFree(d_expand_parent_queue);
		if (d_compact_parent_queue) 	cudaFree(d_compact_parent_queue);
	}


	/**
	 * Initialize from device CSR problem
	 */
	cudaError_t FromDeviceProblem(
		SizeT 		nodes,
		SizeT 		edges,
		VertexId 	*d_column_indices,
		SizeT 		*d_row_offsets,
		VertexId 	*d_source_path = NULL,
		SizeT 		max_expand_queue_size = -1,
		SizeT 		max_compact_queue_size = -1)
	{
		this->nodes 			= nodes;
		this->edges 			= edges;
		this->d_column_indices 	= d_column_indices;
		this->d_row_offsets 	= d_row_offsets;
		this->d_source_path 	= d_source_path;

		this->max_expand_queue_size 	= (max_expand_queue_size > 0) ?
			max_expand_queue_size : 													// Queue-size override
			((long long) edges) * (DEFAULT_QUEUE_PADDING_PERCENT + 100) / 100;			// Use default queue size

		this->max_compact_queue_size 	= (max_compact_queue_size > 0) ?
			max_compact_queue_size : 													// Queue-size override
			((long long) nodes) * (DEFAULT_QUEUE_PADDING_PERCENT + 100) / 100;			// Use default queue size

		printf("Expand queue size: %d, Compact queue size: %d\n", this->max_expand_queue_size, this->max_compact_queue_size);

		return cudaSuccess;
	}

	/**
	 * Initialize from host CSR problem
	 */
	cudaError_t FromHostProblem(
		SizeT 		nodes,
		SizeT 		edges,
		VertexId 	*h_column_indices,
		SizeT 		*h_row_offsets,
		SizeT 		max_expand_queue_size = -1,
		SizeT 		max_compact_queue_size = -1)
	{
		cudaError_t retval = cudaSuccess;

		this->nodes = nodes;
		this->edges = edges;

		this->max_expand_queue_size 	= (max_expand_queue_size > 0) ?
			max_expand_queue_size : 													// Queue-size override
			((long long) edges) * (DEFAULT_QUEUE_PADDING_PERCENT + 100) / 100;			// Use default queue size

		this->max_compact_queue_size 	= (max_compact_queue_size > 0) ?
			max_compact_queue_size : 													// Queue-size override
			((long long) nodes) * (DEFAULT_QUEUE_PADDING_PERCENT + 100) / 100;			// Use default queue size

		printf("Expand queue size: %d, Compact queue size: %d\n", this->max_expand_queue_size, this->max_compact_queue_size);

		do {
			// Allocate and initialize d_column_indices
			if (retval = util::B40CPerror(cudaMalloc((void**) &d_column_indices, edges * sizeof(VertexId)),
				"BfsCsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaMemcpy(d_column_indices, h_column_indices, edges * sizeof(VertexId), cudaMemcpyHostToDevice),
				"BfsCsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

			// Allocate and initialize d_row_offsets
			if (retval = util::B40CPerror(cudaMalloc((void**) &d_row_offsets, (nodes + 1) * sizeof(SizeT)),
				"BfsCsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaMemcpy(d_row_offsets, h_row_offsets, (nodes + 1) * sizeof(SizeT), cudaMemcpyHostToDevice),
				"BfsCsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}


	/**
	 * Performs any initialization work needed for this problem type.  Must be called
	 * prior to each search
	 */
	cudaError_t Reset()
	{
		cudaError_t retval = cudaSuccess;

		do {
			//
			// Allocate ancillary storage if necessary
			//

			// Allocate d_source_path if necessary
			if (!d_source_path) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_source_path, nodes * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_source_path failed", __FILE__, __LINE__)) break;
			}

			// Allocate d_collision_cache if necessary
			int bitmask_bytes 			= ((nodes * sizeof(CollisionMask)) + 8 - 1) / 8;					// round up to the nearest CollisionMask
			int bitmask_elements		= bitmask_bytes * sizeof(CollisionMask);
			if (!d_collision_cache) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_collision_cache, bitmask_bytes),
					"BfsCsrProblem cudaMalloc d_collision_cache failed", __FILE__, __LINE__)) break;
			}

			// Allocate queues if necessary
			if (!d_expand_queue) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_expand_queue, max_expand_queue_size * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_expand_queue failed", __FILE__, __LINE__)) break;
			}
			if (!d_compact_queue) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_compact_queue, max_compact_queue_size * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_compact_queue failed", __FILE__, __LINE__)) break;
			}

			if (MARK_PARENTS) {
				// Allocate parent vertex queues if necessary
				if (!d_expand_parent_queue) {
					if (retval = util::B40CPerror(cudaMalloc((void**) &d_expand_parent_queue, max_expand_queue_size * sizeof(VertexId)),
						"BfsCsrProblem cudaMalloc d_expand_parent_queue failed", __FILE__, __LINE__)) break;
				}
				if (!d_compact_parent_queue) {
					if (retval = util::B40CPerror(cudaMalloc((void**) &d_compact_parent_queue, max_compact_queue_size * sizeof(VertexId)),
						"BfsCsrProblem cudaMalloc d_compact_parent_queue failed", __FILE__, __LINE__)) break;
				}
			}


			// Allocate d_keep if necessary
			if (!d_keep) {
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_keep, max_expand_queue_size * sizeof(ValidFlag)),
					"BfsCsrProblem cudaMalloc d_keep failed", __FILE__, __LINE__)) break;
			}


			//
			// Initialize ancillary storage
			//

			int memset_block_size 		= 256;
			int memset_grid_size_max 	= 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_source_path elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (nodes + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size>>>(
				d_source_path, -1, nodes);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

			// Initialize d_collision_cache elements to 0
			memset_grid_size = B40C_MIN(memset_grid_size_max, (bitmask_elements + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<CollisionMask><<<memset_grid_size, memset_block_size>>>(
				d_collision_cache, 0, bitmask_elements);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

		} while (0);

		return retval;
	}
};


} // namespace bfs
} // namespace b40c

