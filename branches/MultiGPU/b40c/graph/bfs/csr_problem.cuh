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
 * GPU CSR storage management structure for BFS problem data
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>

#include <b40c/graph/bfs/problem_type.cuh>

#include <vector>

namespace b40c {
namespace graph {
namespace bfs {


/**
 * CSR storage management structure for BFS problems.  
 */
template <
	typename _VertexId,
	typename _SizeT,
	bool MARK_PARENTS>		// Whether to mark parents vs mark distance-from-source
struct CsrProblem
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	static const int DEFAULT_QUEUE_SIZING = 1.20;

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


	//---------------------------------------------------------------------
	// Helper structures
	//---------------------------------------------------------------------

	/**
	 * Graph slice per GPU
	 */
	struct GraphSlice
	{
		// GPU index
		int 			gpu;

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

		// Number of nodes and edges in slice
		VertexId		nodes;
		SizeT			edges;

		// CUDA stream to use for processing this slice
		cudaStream_t 	stream;

		/**
		 * Constructor
		 */
		GraphSlice(cudaStream_t stream) :
			d_column_indices(NULL),
			d_row_offsets(NULL),
			d_source_path(NULL),
			d_collision_cache(NULL),
			d_keep(NULL),
			d_expand_queue(NULL),
			d_compact_queue(NULL),
			d_expand_parent_queue(NULL),
			d_compact_parent_queue(NULL),
			stream(stream)
		{}

		/**
		 * Destructor
		 */
		virtual ~GpuSlice()
		{
			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"GpuSlice cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Free pointers
			if (d_column_indices) 			cudaFree(d_column_indices);
			if (d_row_offsets) 				cudaFree(d_row_offsets);
			if (d_source_path) 				cudaFree(d_source_path);
			if (d_collision_cache) 			cudaFree(d_collision_cache);
			if (d_keep) 					cudaFree(d_keep);
			if (d_expand_queue) 			cudaFree(d_expand_queue);
			if (d_compact_queue) 			cudaFree(d_compact_queue);
			if (d_expand_parent_queue) 		cudaFree(d_expand_parent_queue);
			if (d_compact_parent_queue) 	cudaFree(d_compact_parent_queue);

	        // Destroy stream
			if (util::B40CPerror(cudaStreamDestroy(stream),
				"GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__)) exit(1);
		}
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Number of GPUS to be sliced over
	int							num_gpus;

	// Maximum size factor (in terms of total edges) of the queues
	double 						queue_sizing;

	// Size of the graph
	SizeT 						nodes;
	SizeT						edges;

	// GPU graph slices
	std::vector<GraphSlice*> 	graph_slices;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	CsrProblem() :
		num_gpus(0),
		queue_sizing(DEFAULT_QUEUE_SIZING),
		nodes(0),
		edges(0)
	{}


	/**
	 * Destructor
	 */
	virtual ~CsrProblem()
	{
		// Cleanup graph slices on the heap
		for (std::vector<GraphSlice*>::iterator itr = graph_slices.begin(); itr != graph_slices.end(); itr++) {
			if (*itr) delete (*itr);
		}
	}


	/**
	 * Returns index of the gpu that owns the neighbor list of
	 * the specified vertex
	 */
	template <typename VertexId>
	int GpuIndex(VertexId vertex)
	{
		return vertex & (num_gpus - 1);
	}


	/**
	 * Returns the row within a gpu's GraphSlice row_offsets vector
	 * for the specified vertex
	 */
	template <typename VertexId>
	VertexId GraphSliceRow(VertexId vertex)
	{
		return vertex / num_gpus;
	}


	/**
	 * Extract into a single host vector the BFS results disseminated across
	 * all GPUs
	 */
	cudaError_t ExtractResults(VertexId *h_source_path)
	{
		cudaError_t retval = cudaSuccess;

		VertexId **gpu_source_paths = new VertexId*[num_gpus];

		// Copy out
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Allocate and copy out
			gpu_source_paths[gpu] = new VertexId[graph_slices[gpu]->nodes];

			if (retval = util::B40CPerror(cudaMemcpy(
					gpu_source_paths[gpu],
					sizeof(VertexId) * graph_slices[gpu]->nodes,
					cudaMemcpyDeviceToHost),
				"CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
		}

		// Combine
		for (VertexId node = 0; node < nodes; node++) {
			int gpu = GpuIndex(node);
			VertexId slice_row = GraphSliceRow(node);
			h_source_path[node] = gpu_source_paths[gpu][slice_row];
		}

		// Clean up
		for (int gpu = 0; gpu < num_gpus; gpu++) {
			if (gpu_source_paths[gpu]) delete gpu_source_paths[gpu];
		}
		delete gpu_source_paths;

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

		this->queue_sizing = (queue_sizing <= 0.0) ?
			DEFAULT_QUEUE_SIZING :
			queue_sizing;

		// Create GPU slices
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Create stream
			cudaStream_t stream;
			if (util::B40CPerror(cudaStreamCreate(&stream),
				"CsrProblem cudaStreamCreate failed", __FILE__, __LINE__)) exit(1);

			// Create slice
			graph_slices.push_back(new GraphSlice(stream));
		}

		// Count up nodes and edges for each gpu
		for (VertexId node = 0; node < nodes; node++) {
			int gpu = GpuIndex(node);
			graph_slices[gpu]->nodes++;
			graph_slices[gpu]->edges += h_row_offsets[node + 1] - h_row_offsets[node];
		}

		// Allocate data structures for gpu on host
		SizeT **slice_row_offsets 			= new SizeT*[num_gpus];
		VertexId **slice_column_indices 	= new VertexId*[num_gpus];
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			printf("GPU %d gets %d vertices and %d edges\n",
				gpu, graph_slices[gpu]->nodes, graph_slices[gpu]->edges);
			fflush(stdout);

			slice_row_offsets[gpu] = new SizeT[graph_slices[gpu]->nodes + 1];
			slice_row_offsets[gpu][0] = 0;

			slice_column_indices[gpu] = new VertexId[graph_slices[gpu]->edges];

			// Reset for construction
			graph_slices[gpu]->edges = 0;
		}

		printf("Done allocating gpu data structures\n");
		fflush(stdout);

		// Construct data structures for gpus on host
		for (VertexId node = 0; node < nodes; node++) {

			int gpu 				= GpuIndex(node);
			VertexId slice_row 		= GraphSliceRow(node);
			SizeT row_edges			= h_row_offsets[node + 1] - h_row_offsets[node];

			memcpy(
				slice_column_indices[gpu] + slice_row_offsets[gpu][slice_row],
				h_column_indices + h_row_offsets[node],
				row_edges * sizeof(VertexId));

			graph_slices[gpu]->edges += row_edges;
			slice_row_offsets[gpu][slice_row + 1] = graph_slices[gpu]->edges;
		}

		printf("Done constructing gpu data structures\n");
		fflush(stdout);

		// Initialize data structures on GPU
		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Allocate and initialize d_row_offsets: copy and adjust by gpu offset
			if (retval = util::B40CPerror(cudaMalloc(
					(void**) &graph_slices[gpu]->d_row_offsets,
					(graph_slices[gpu]->nodes + 1) * sizeof(SizeT)),
				"CsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

			if (retval = util::B40CPerror(cudaMemcpy(
					graph_slices[gpu]->d_row_offsets,
					slice_row_offsets[gpu],
					(graph_slices[gpu]->nodes + 1) * sizeof(SizeT),
					cudaMemcpyHostToDevice),
				"CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;

			// Allocate and initialize d_column_indices
			if (retval = util::B40CPerror(cudaMalloc(
					(void**) &graph_slices[gpu]->d_column_indices,
					graph_slices[gpu]->edges * sizeof(VertexId)),
				"CsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

			if (retval = util::B40CPerror(cudaMemcpy(
					graph_slices[gpu]->d_column_indices,
					slice_column_indices[gpu],
					graph_slices[gpu]->edges * sizeof(VertexId),
					cudaMemcpyHostToDevice),
				"CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

			// Cleanup host construction arrays
			if (slice_row_offsets[gpu]) delete slice_row_offsets[gpu];
			if (slice_column_indices[gpu]) delete slice_column_indices[gpu];

			delete slice_row_offsets;
			delete slice_column_indices;
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
				"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			//
			// Allocate ancillary storage if necessary
			//

			// Allocate d_source_path if necessary
			if (!graph_slices[gpu]->d_source_path) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_source_path,
						graph_slices[gpu]->nodes * sizeof(VertexId)),
					"CsrProblem cudaMalloc d_source_path failed", __FILE__, __LINE__)) break;
			}

			// Allocate d_collision_cache for the entire graph if necessary
			int bitmask_bytes 			= ((nodes * sizeof(CollisionMask)) + 8 - 1) / 8;					// round up to the nearest CollisionMask
			int bitmask_elements		= bitmask_bytes * sizeof(CollisionMask);
			if (!graph_slices[gpu]->d_collision_cache) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_collision_cache,
						bitmask_bytes),
					"CsrProblem cudaMalloc d_collision_cache failed", __FILE__, __LINE__)) break;
			}

			// Allocate queues if necessary
			SizeT queue_elements = double(graph_slices[gpu]->edges) * queue_sizing;
			printf("GPU %d queue size: %d\n", gpu, queue_elements);
			fflush(stdout);

			if (!graph_slices[gpu]->d_expand_queue) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_expand_queue,
						queue_elements * sizeof(VertexId)),
					"CsrProblem cudaMalloc d_expand_queue failed", __FILE__, __LINE__)) break;
			}
			if (!graph_slices[gpu]->d_compact_queue) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_compact_queue,
						queue_elements * sizeof(VertexId)),
					"CsrProblem cudaMalloc d_compact_queue failed", __FILE__, __LINE__)) break;
			}

			if (MARK_PARENTS) {
				// Allocate parent vertex queues if necessary
				if (!graph_slices[gpu]->d_expand_parent_queue) {
					if (retval = util::B40CPerror(
							cudaMalloc((void**) &graph_slices[gpu]->d_expand_parent_queue,
							queue_elements * sizeof(VertexId)),
						"CsrProblem cudaMalloc d_expand_parent_queue failed", __FILE__, __LINE__)) break;
				}
				if (!graph_slices[gpu]->d_compact_parent_queue) {
					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &graph_slices[gpu]->d_compact_parent_queue,
							queue_elements * sizeof(VertexId)),
						"CsrProblem cudaMalloc d_compact_parent_queue failed", __FILE__, __LINE__)) break;
				}
			}

			// Allocate d_keep if necessary
			if (!graph_slices[gpu]->d_keep) {
				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_keep,
						queue_elements * sizeof(ValidFlag)),
					"CsrProblem cudaMalloc d_keep failed", __FILE__, __LINE__)) break;
			}


			//
			// Initialize source paths and collision mask cache
			//

			int memset_block_size 		= 256;
			int memset_grid_size_max 	= 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_source_path elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
				graph_slices[gpu]->d_source_path,
				-1,
				graph_slices[gpu]->nodes);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

			// Initialize d_collision_cache elements to 0
			memset_grid_size = B40C_MIN(memset_grid_size_max, (bitmask_elements + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<CollisionMask><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
				graph_slices[gpu]->d_collision_cache,
				0,
				bitmask_elements);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

		}

		return retval;
	}
};



} // namespace bfs
} // namespace graph
} // namespace b40c
