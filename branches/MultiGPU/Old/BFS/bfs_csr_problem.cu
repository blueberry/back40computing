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

#define B40C_DEFAULT_QUEUE_SIZING 1.15

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


	static const int MAX_GPUS = 4;

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Size of the graph
	SizeT 			nodes;
	SizeT			edges;

	// Standard CSR device storage arrays
	VertexId 		*d_column_indices[MAX_GPUS];
	SizeT 			*d_row_offsets[MAX_GPUS];
	VertexId 		*d_source_path[MAX_GPUS];				// Can be used for source distance or parent pointer

	// Best-effort (bit) mask for keeping track of which vertices we've seen so far
	CollisionMask 	*d_collision_cache[MAX_GPUS];
	
	// Frontier queues
	VertexId 		*d_expand_queue[MAX_GPUS];
	VertexId 		*d_compact_queue[MAX_GPUS];

	// Optional queues for tracking parent vertices
	VertexId 		*d_expand_parent_queue[MAX_GPUS];
	VertexId 		*d_compact_parent_queue[MAX_GPUS];

	// Vector of valid flags for elements in the frontier queue
	ValidFlag 		*d_keep[MAX_GPUS];

	// Maximum size of the queues
	SizeT 			max_expand_queue_size;
	SizeT 			max_compact_queue_size;

	// Actual number of GPUS
	int				num_gpus;
	VertexId		gpu_vertex_range[MAX_GPUS + 1];
	SizeT			gpu_edge_range[MAX_GPUS + 1];


	/**
	 * Streams
	 */
	cudaStream_t stream[MAX_GPUS];


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	BfsCsrProblem() :
		nodes(0),
		edges(0),
		max_expand_queue_size(0),
		max_compact_queue_size(0),
		num_gpus(0)
	{
		for (int i = 0; i < MAX_GPUS; i++) {
			d_column_indices[i] = NULL;
			d_row_offsets[i] = NULL;
			d_source_path[i] = NULL;
			d_collision_cache[i] = NULL;
			d_keep[i] = NULL;
			d_expand_queue[i] = NULL;
			d_compact_queue[i] = NULL;
			d_expand_parent_queue[i] = NULL;
			d_compact_parent_queue[i] = NULL;
		}
	}


	/**
     * Destructor
     */
    virtual ~BfsCsrProblem()
    {
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Free pointers
			if (d_column_indices[gpu]) 				cudaFree(d_column_indices[gpu]);
			if (d_row_offsets[gpu]) 				cudaFree(d_row_offsets[gpu]);
			if (d_source_path[gpu]) 				cudaFree(d_source_path[gpu]);
			if (d_collision_cache[gpu]) 			cudaFree(d_collision_cache[gpu]);
			if (d_keep[gpu]) 						cudaFree(d_keep[gpu]);
			if (d_expand_queue[gpu]) 				cudaFree(d_expand_queue[gpu]);
			if (d_compact_queue[gpu]) 				cudaFree(d_compact_queue[gpu]);
			if (d_expand_parent_queue[gpu]) 		cudaFree(d_expand_parent_queue[gpu]);
			if (d_compact_parent_queue[gpu]) 		cudaFree(d_compact_parent_queue[gpu]);

	        // Destroy stream
			if (util::B40CPerror(cudaStreamDestroy(stream[gpu]),
				"MultiGpuBfsEnactor cudaStreamDestroy failed", __FILE__, __LINE__)) exit(1);
		}
	}

	/**
	 * Returns the number of bits to shift a vertex ID to get a gpu_id
	 */
	template <typename VertexId>
	int ShiftBits(VertexId nodes, int num_gpus)
	{
		int retval = 0;
		VertexId max_vertex = nodes - 1;
		while (max_vertex >= num_gpus) {
			max_vertex >>= 1;
			retval++;
		}
		return retval;
	}

	/**
	 *
	 */
	cudaError_t CombineResults(VertexId *h_source_path)
	{
		cudaError_t retval = cudaSuccess;

		for (int gpu = 0; gpu < num_gpus; gpu++) {

			VertexId gpu_vertices = gpu_vertex_range[gpu + 1] - gpu_vertex_range[gpu];

			if (retval = util::B40CPerror(cudaMemcpy(
					h_source_path + gpu_vertex_range[gpu],
					d_source_path[gpu],
					gpu_vertices * sizeof(SizeT),
					cudaMemcpyDeviceToHost),
				"BfsCsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
		}

		return retval;
	}


	/**
	 * Initialize from host CSR problem
	 */
	cudaError_t FromHostProblem(
		SizeT 		nodes,
		SizeT 		edges,
		VertexId 	*h_column_indices,
		SizeT 		*h_row_offsets,
		double 		queue_sizing,
		int 		num_gpus)
	{
		cudaError_t retval 				= cudaSuccess;
		this->nodes						= nodes;
		this->edges 					= edges;
		this->num_gpus 					= num_gpus;
		this->max_expand_queue_size 	= (queue_sizing > 0.0) ?
											queue_sizing * double(edges) : 							// Queue-size override
											double(B40C_DEFAULT_QUEUE_SIZING) * double(edges);		// Use default queue size
		this->max_compact_queue_size 	= max_expand_queue_size;

		printf("Expand queue size: %d, Compact queue size: %d\n", this->max_expand_queue_size, this->max_compact_queue_size);

		int shift_bits = ShiftBits(nodes, num_gpus);
		SizeT *h_gpu_row_offsets = NULL;
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"BfsCsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Create stream
			if (util::B40CPerror(cudaStreamCreate(&stream[gpu]),
				"BfsCsrProblem cudaStreamCreate failed", __FILE__, __LINE__)) exit(1);

			// Compute GPU offsets
			gpu_vertex_range[gpu] 		= gpu << shift_bits;
			gpu_vertex_range[gpu + 1]	= B40C_MIN(nodes, (gpu + 1) << shift_bits);
			VertexId gpu_vertices 		= gpu_vertex_range[gpu + 1] - gpu_vertex_range[gpu];

			gpu_edge_range[gpu]			= h_row_offsets[gpu_vertex_range[gpu]];
			gpu_edge_range[gpu + 1]		= h_row_offsets[gpu_vertex_range[gpu + 1]];
			SizeT gpu_edges				= gpu_edge_range[gpu + 1] - gpu_edge_range[gpu];

			printf("GPU %d gets %d vertices [%d,%d) and %d edges[%d,%d)\n",
				gpu,
				gpu_vertices, gpu_vertex_range[gpu], gpu_vertex_range[gpu + 1],
				gpu_edges, gpu_edge_range[gpu], gpu_edge_range[gpu + 1]);

			// Allocate and initialize d_column_indices
			if (retval = util::B40CPerror(cudaMalloc(
					(void**) &d_column_indices[gpu],
					gpu_edges * sizeof(VertexId)),
				"BfsCsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaMemcpy(
					d_column_indices[gpu],
					h_column_indices + gpu_edge_range[gpu],
					gpu_edges * sizeof(VertexId),
					cudaMemcpyHostToDevice),
				"BfsCsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

			// Allocate and initialize d_row_offsets: copy and adjust by gpu offset
			if (h_gpu_row_offsets) {
				free(h_gpu_row_offsets);
			}
			h_gpu_row_offsets = (SizeT *) malloc((gpu_vertices + 1) * sizeof(SizeT));
			for (int row = 0; row < gpu_vertices + 1; row++) {
				h_gpu_row_offsets[row] = h_row_offsets[gpu_vertex_range[gpu] + row] - gpu_edge_range[gpu];
			}

			if (retval = util::B40CPerror(cudaMalloc(
					(void**) &d_row_offsets[gpu],
					(gpu_vertices + 1) * sizeof(SizeT)),
				"BfsCsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaMemcpy(
					d_row_offsets[gpu],
					h_gpu_row_offsets,
					(gpu_vertices + 1) * sizeof(SizeT),
					cudaMemcpyHostToDevice),
				"BfsCsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
		};

		if (h_gpu_row_offsets) free(h_gpu_row_offsets);

		return retval;
	}


	/**
	 * Performs any initialization work needed for this problem type.  Must be called
	 * prior to each search
	 */
	cudaError_t Reset()
	{
		cudaError_t retval = cudaSuccess;

		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"BfsCsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			//
			// Allocate ancillary storage if necessary
			//
			VertexId gpu_vertices = gpu_vertex_range[gpu + 1] - gpu_vertex_range[gpu];

			// Allocate d_source_path if necessary
			if (!d_source_path[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_source_path[gpu],
						gpu_vertices * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_source_path failed", __FILE__, __LINE__)) break;
			}

			// Allocate d_collision_cache for the entire graph if necessary
			int bitmask_bytes 			= ((nodes * sizeof(CollisionMask)) + 8 - 1) / 8;					// round up to the nearest CollisionMask
			int bitmask_elements		= bitmask_bytes * sizeof(CollisionMask);
			if (!d_collision_cache[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_collision_cache[gpu],
						bitmask_bytes),
					"BfsCsrProblem cudaMalloc d_collision_cache failed", __FILE__, __LINE__)) break;
			}

			// Allocate queues if necessary
			if (!d_expand_queue[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_expand_queue[gpu],
						max_expand_queue_size * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_expand_queue failed", __FILE__, __LINE__)) break;
			}
			if (!d_compact_queue[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_compact_queue[gpu],
						max_compact_queue_size * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_compact_queue failed", __FILE__, __LINE__)) break;
			}

			if (MARK_PARENTS) {
				// Allocate parent vertex queues if necessary
				if (!d_expand_parent_queue[gpu]) {
					if (retval = util::B40CPerror(
							cudaMalloc((void**) &d_expand_parent_queue[gpu],
							max_expand_queue_size * sizeof(VertexId)),
						"BfsCsrProblem cudaMalloc d_expand_parent_queue failed", __FILE__, __LINE__)) break;
				}
				if (!d_compact_parent_queue[gpu]) {
					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &d_compact_parent_queue[gpu],
							max_compact_queue_size * sizeof(VertexId)),
						"BfsCsrProblem cudaMalloc d_compact_parent_queue failed", __FILE__, __LINE__)) break;
				}
			}


			// Allocate d_keep if necessary
			if (!d_keep[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_keep[gpu],
						max_expand_queue_size * sizeof(ValidFlag)),
					"BfsCsrProblem cudaMalloc d_keep failed", __FILE__, __LINE__)) break;
			}


			//
			// Initialize ancillary storage
			//

			int memset_block_size 		= 256;
			int memset_grid_size_max 	= 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_source_path elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (gpu_vertices + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size, 0, stream[gpu]>>>(
				d_source_path[gpu], -1, nodes);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

			// Initialize d_collision_cache elements to 0
			memset_grid_size = B40C_MIN(memset_grid_size_max, (bitmask_elements + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<CollisionMask>
				<<<memset_grid_size, memset_block_size, 0, stream[gpu]>>>(
					d_collision_cache[gpu], 0, bitmask_elements);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

		}

		return retval;
	}
};

#undef B40C_DEFAULT_QUEUE_SIZING


} // namespace bfs
} // namespace b40c

