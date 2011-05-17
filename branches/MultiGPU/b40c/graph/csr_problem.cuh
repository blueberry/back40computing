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

#define B40C_DEFAULT_QUEUE_SIZING 1.25

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

	// Maximum size factor (in terms of total edges) of the queues
	double 			queue_sizing;

	// Actual number of GPUS
	int				num_gpus;
	VertexId		gpu_nodes[MAX_GPUS + 1];
	SizeT			gpu_edges[MAX_GPUS + 1];


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
		queue_sizing(1.0),
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

	template <typename VertexId>
	int Gpu(VertexId vertex)
	{
		return vertex & (num_gpus - 1);
	}

	template <typename VertexId>
	VertexId GpuRow(VertexId vertex)
	{
		return vertex / num_gpus;
	}

	/**
	 *
	 */
	cudaError_t CombineResults(VertexId *h_source_path)
	{
		cudaError_t retval = cudaSuccess;

		// Copy out
		VertexId* gpu_source_path[MAX_GPUS];
		for (int gpu = 0; gpu < num_gpus; gpu++) {
			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"BfsCsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Allocate and copy out
			gpu_source_path[gpu] = (VertexId *) malloc(sizeof(VertexId) * gpu_nodes[gpu]);
			if (retval = util::B40CPerror(cudaMemcpy(
					gpu_source_path[gpu],
					d_source_path[gpu],
					sizeof(VertexId) * gpu_nodes[gpu],
					cudaMemcpyDeviceToHost),
				"BfsCsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
		}

		// Combine
		for (VertexId node = 0; node < nodes; node++) {

			int 		gpu = Gpu(node);
			VertexId 	gpu_row = GpuRow(node);
			h_source_path[node] = gpu_source_path[gpu][gpu_row];
		}

		// Clean up
		for (int gpu = 0; gpu < num_gpus; gpu++) {
			if (gpu_source_path[gpu]) free(gpu_source_path[gpu]);
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

		if (queue_sizing <= 0.0) {
			queue_sizing = B40C_DEFAULT_QUEUE_SIZING;
		}
		this->queue_sizing = queue_sizing;

		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"BfsCsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Create stream
			if (util::B40CPerror(cudaStreamCreate(&stream[gpu]),
				"BfsCsrProblem cudaStreamCreate failed", __FILE__, __LINE__)) exit(1);

			gpu_nodes[gpu] = 0;
			gpu_edges[gpu] = 0;
		}

		// Count up nodes and edges for each gpu
		for (VertexId node = 0; node < nodes; node++) {
			int gpu = Gpu(node);
			gpu_nodes[gpu]++;
			gpu_edges[gpu] += h_row_offsets[node + 1] - h_row_offsets[node];
		}

		// Allocate data structures for gpus on host
		SizeT* gpu_row_offsets[MAX_GPUS];
		VertexId* gpu_column_indices[MAX_GPUS];
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			printf("GPU %d gets %d vertices and %d edges\n",
				gpu, gpu_nodes[gpu], gpu_edges[gpu]);
			fflush(stdout);

			gpu_row_offsets[gpu] = (SizeT*) malloc(sizeof(SizeT) * (gpu_nodes[gpu] + 1));
			gpu_row_offsets[gpu][0] = 0;

			gpu_column_indices[gpu] = (VertexId*) malloc(sizeof(VertexId) * gpu_edges[gpu]);

			// Reset for construction
			gpu_edges[gpu] = 0;
		}

		printf("Done allocating gpu data structures\n");
		fflush(stdout);

		// Construct data structures for gpus on host
		for (VertexId node = 0; node < nodes; node++) {

			int 		gpu = Gpu(node);
			VertexId 	gpu_row = GpuRow(node);
			SizeT 		row_edges = h_row_offsets[node + 1] - h_row_offsets[node];

			memcpy(
				gpu_column_indices[gpu] + gpu_row_offsets[gpu][gpu_row],
				h_column_indices + h_row_offsets[node],
				row_edges * sizeof(VertexId));

			gpu_edges[gpu] += row_edges;
			gpu_row_offsets[gpu][gpu_row + 1] = gpu_edges[gpu];
		}

		printf("Done constructing gpu data structures\n");
		fflush(stdout);

		// Initialize data structures on GPU
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"BfsCsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Allocate and initialize d_row_offsets: copy and adjust by gpu offset
			if (retval = util::B40CPerror(cudaMalloc(
					(void**) &d_row_offsets[gpu],
					(gpu_nodes[gpu] + 1) * sizeof(SizeT)),
				"BfsCsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaMemcpy(
					d_row_offsets[gpu],
					gpu_row_offsets[gpu],
					(gpu_nodes[gpu] + 1) * sizeof(SizeT),
					cudaMemcpyHostToDevice),
				"BfsCsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

			// Allocate and initialize d_column_indices
			if (retval = util::B40CPerror(cudaMalloc(
					(void**) &d_column_indices[gpu],
					gpu_edges[gpu] * sizeof(VertexId)),
				"BfsCsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;
			if (retval = util::B40CPerror(cudaMemcpy(
					d_column_indices[gpu],
					gpu_column_indices[gpu],
					gpu_edges[gpu] * sizeof(VertexId),
					cudaMemcpyHostToDevice),
				"BfsCsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

			if (gpu_row_offsets[gpu]) free(gpu_row_offsets[gpu]);
			if (gpu_column_indices[gpu]) free(gpu_column_indices[gpu]);
		};

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

			// Allocate d_source_path if necessary
			if (!d_source_path[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_source_path[gpu],
						gpu_nodes[gpu] * sizeof(VertexId)),
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
			SizeT queue_elements = double(gpu_edges[gpu]) * queue_sizing;
			printf("GPU %d queue size: %d\n", gpu, queue_elements);
			fflush(stdout);

			if (!d_expand_queue[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_expand_queue[gpu],
						queue_elements * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_expand_queue failed", __FILE__, __LINE__)) break;
			}
			if (!d_compact_queue[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_compact_queue[gpu],
						queue_elements * sizeof(VertexId)),
					"BfsCsrProblem cudaMalloc d_compact_queue failed", __FILE__, __LINE__)) break;
			}

			if (MARK_PARENTS) {
				// Allocate parent vertex queues if necessary
				if (!d_expand_parent_queue[gpu]) {
					if (retval = util::B40CPerror(
							cudaMalloc((void**) &d_expand_parent_queue[gpu],
							queue_elements * sizeof(VertexId)),
						"BfsCsrProblem cudaMalloc d_expand_parent_queue failed", __FILE__, __LINE__)) break;
				}
				if (!d_compact_parent_queue[gpu]) {
					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &d_compact_parent_queue[gpu],
							queue_elements * sizeof(VertexId)),
						"BfsCsrProblem cudaMalloc d_compact_parent_queue failed", __FILE__, __LINE__)) break;
				}
			}


			// Allocate d_keep if necessary
			if (!d_keep[gpu]) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &d_keep[gpu],
						queue_elements * sizeof(ValidFlag)),
					"BfsCsrProblem cudaMalloc d_keep failed", __FILE__, __LINE__)) break;
			}


			//
			// Initialize ancillary storage
			//

			int memset_block_size 		= 256;
			int memset_grid_size_max 	= 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_source_path elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (gpu_nodes[gpu] + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VertexId>
					<<<memset_grid_size, memset_block_size, 0, stream[gpu]>>>(
				d_source_path[gpu],
				-1,
				gpu_nodes[gpu]);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

			// Initialize d_collision_cache elements to 0
			memset_grid_size = B40C_MIN(memset_grid_size_max, (bitmask_elements + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<CollisionMask>
					<<<memset_grid_size, memset_block_size, 0, stream[gpu]>>>(
				d_collision_cache[gpu],
				0,
				bitmask_elements);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

		}

		return retval;
	}
};

#undef B40C_DEFAULT_QUEUE_SIZING


} // namespace bfs
} // namespace b40c

