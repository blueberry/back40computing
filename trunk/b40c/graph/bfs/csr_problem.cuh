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
#include <b40c/util/ping_pong_storage.cuh>

#include <b40c/graph/bfs/problem_type.cuh>

#include <vector>

namespace b40c {
namespace graph {
namespace bfs {


/**
 * Enumeration for out-of-core frontier queue configurations
 */
enum FrontierType {
	VERTEX_FRONTIERS,	// O(n) ping/pong frontiers
	EDGE_FRONTIERS,		// O(m) ping/pong frontiers
	MIXED_FRONTIERS,		// O(n) ping frontier, O(m) pong frontier
};



/**
 * CSR storage management structure for BFS problems.  
 */
template <
	typename _VertexId,
	typename _SizeT,
	bool MARK_PREDECESSORS>		// Whether to mark predecessors (vs. mark distance from source)
struct CsrProblem
{
	//---------------------------------------------------------------------
	// Typedefs and constants
	//---------------------------------------------------------------------

	static const float DEFAULT_QUEUE_SIZING;
	static const int LOG_MAX_GPUS				= 2;

	typedef ProblemType<
		_VertexId,				// VertexId
		_SizeT,					// SizeT
		unsigned char,			// VisitedMask
		unsigned char, 			// ValidFlag
		MARK_PREDECESSORS,		// MARK_PREDECESSORS
		LOG_MAX_GPUS>			// LOG_MAX_GPUS
			ProblemType;

	typedef typename ProblemType::VertexId 			VertexId;
	typedef typename ProblemType::SizeT				SizeT;
	typedef typename ProblemType::VisitedMask 		VisitedMask;
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
		VertexId 		*d_labels;				// Can be used for source distance or predecessor pointer

		// Best-effort (bit) mask for keeping track of which vertices we've seen so far
		VisitedMask 	*d_visited_mask;

		// Frontier queues (keys track work, values optionally track predecessors)
		util::PingPongStorage<VertexId, VertexId> 	frontier_queues;
		SizeT 										ping_elements;
		SizeT 										pong_elements;

		VertexId		*d_multigpu_vqueue;
		SizeT 			multigpu_vqueue_elements;

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
		GraphSlice(int gpu, cudaStream_t stream) :
			gpu(gpu),
			d_column_indices(NULL),
			d_row_offsets(NULL),
			d_labels(NULL),
			d_visited_mask(NULL),
			d_keep(NULL),
			ping_elements(0),
			pong_elements(0),
			d_multigpu_vqueue(NULL),
			multigpu_vqueue_elements(0),
			nodes(0),
			edges(0),
			stream(stream)
		{}

		/**
		 * Destructor
		 */
		virtual ~GraphSlice()
		{
			// Set device
			util::B40CPerror(cudaSetDevice(gpu), "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

			// Free pointers
			if (d_column_indices) 				util::B40CPerror(cudaFree(d_column_indices), "GpuSlice cudaFree d_column_indices failed", __FILE__, __LINE__);
			if (d_row_offsets) 					util::B40CPerror(cudaFree(d_row_offsets), "GpuSlice cudaFree d_row_offsets failed", __FILE__, __LINE__);
			if (d_labels) 						util::B40CPerror(cudaFree(d_labels), "GpuSlice cudaFree d_labels failed", __FILE__, __LINE__);
			if (d_visited_mask) 				util::B40CPerror(cudaFree(d_visited_mask), "GpuSlice cudaFree d_visited_mask failed", __FILE__, __LINE__);
			if (d_keep) 						util::B40CPerror(cudaFree(d_keep), "GpuSlice cudaFree d_keep failed", __FILE__, __LINE__);
			if (frontier_queues.d_keys[0]) 		util::B40CPerror(cudaFree(frontier_queues.d_keys[0]), "GpuSlice cudaFree frontier_queues.d_keys[0] failed", __FILE__, __LINE__);
			if (frontier_queues.d_keys[1]) 		util::B40CPerror(cudaFree(frontier_queues.d_keys[1]), "GpuSlice cudaFree frontier_queues.d_keys[1] failed", __FILE__, __LINE__);
			if (frontier_queues.d_values[0]) 	util::B40CPerror(cudaFree(frontier_queues.d_values[0]), "GpuSlice cudaFree frontier_queues.d_values[0] failed", __FILE__, __LINE__);
			if (frontier_queues.d_values[1]) 	util::B40CPerror(cudaFree(frontier_queues.d_values[1]), "GpuSlice cudaFree frontier_queues.d_values[1] failed", __FILE__, __LINE__);
			if (d_multigpu_vqueue) 				util::B40CPerror(cudaFree(d_multigpu_vqueue), "GpuSlice cudaFree d_multigpu_vqueue failed", __FILE__, __LINE__);

	        // Destroy stream
			if (stream) {
				util::B40CPerror(cudaStreamDestroy(stream), "GpuSlice cudaStreamDestroy failed", __FILE__, __LINE__);
			}
		}
	};


	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

	// Number of GPUS to be sliced over
	int							num_gpus;

	// Size of the graph
	SizeT 						nodes;
	SizeT						edges;

	// Set of graph slices (one for each GPU)
	std::vector<GraphSlice*> 	graph_slices;


	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

	/**
	 * Constructor
	 */
	CsrProblem() :
		num_gpus(0),
		nodes(0),
		edges(0),
		uneven(false)
	{}


	/**
	 * Destructor
	 */
	virtual ~CsrProblem()
	{
		// Cleanup graph slices on the heap
		for (typename std::vector<GraphSlice*>::iterator itr = graph_slices.begin();
			itr != graph_slices.end();
			itr++)
		{
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
		if (graph_slices.size() == 1) {

			// Special case for only one GPU, which may be set as with
			// an ordinal other than 0.
			return graph_slices[0]->gpu;

		} else {

			return vertex % num_gpus;
		}
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

		do {
			if (graph_slices.size() == 1) {

				// Set device
				if (util::B40CPerror(cudaSetDevice(graph_slices[0]->gpu),
					"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;;

				// Special case for only one GPU, which may be set as with
				// an ordinal other than 0.
				if (retval = util::B40CPerror(cudaMemcpy(
						h_source_path,
						graph_slices[0]->d_labels,
						sizeof(VertexId) * graph_slices[0]->nodes,
						cudaMemcpyDeviceToHost),
					"CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;

			} else {

				VertexId **gpu_source_paths = new VertexId*[num_gpus];

				// Copy out
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					// Set device
					if (util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu),
						"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;;

					// Allocate and copy out
					gpu_source_paths[gpu] = new VertexId[graph_slices[gpu]->nodes];

					if (retval = util::B40CPerror(cudaMemcpy(
							gpu_source_paths[gpu],
							graph_slices[gpu]->d_labels,
							sizeof(VertexId) * graph_slices[gpu]->nodes,
							cudaMemcpyDeviceToHost),
						"CsrProblem cudaMemcpy d_labels failed", __FILE__, __LINE__)) break;
				}
				if (retval) break;

				// Combine
				for (VertexId node = 0; node < nodes; node++) {
					int gpu = GpuIndex(node);
					VertexId slice_row = GraphSliceRow(node);
					h_source_path[node] = gpu_source_paths[gpu][slice_row];

					switch (h_source_path[node]) {
					case -1:
					case -2:
						break;
					default:
						h_source_path[node] &= ProblemType::VERTEX_ID_MASK;
					};
				}

				// Clean up
				for (int gpu = 0; gpu < num_gpus; gpu++) {
					if (gpu_source_paths[gpu]) delete gpu_source_paths[gpu];
				}
				delete gpu_source_paths;
			}
		} while(0);

		return retval;
	}


	/**
	 * Initialize from host CSR problem
	 */
	cudaError_t FromHostProblem(
		bool		stream_from_host,			// only valid for 1 gpu
		SizeT 		nodes,
		SizeT 		edges,
		VertexId 	*h_column_indices,
		SizeT 		*h_row_offsets,
		int 		num_gpus)
	{
		cudaError_t retval 				= cudaSuccess;
		this->nodes						= nodes;
		this->edges 					= edges;
		this->num_gpus 					= num_gpus;

		do {
			if (num_gpus <= 1) {

				// Create a single GPU slice for the currently-set gpu
				int gpu;
				if (retval = util::B40CPerror(cudaGetDevice(&gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
				graph_slices.push_back(new GraphSlice(gpu, 0));
				graph_slices[0]->nodes = nodes;
				graph_slices[0]->edges = edges;

				if (stream_from_host) {

					// Map the pinned graph pointers into device pointers
					if (retval = util::B40CPerror(cudaHostGetDevicePointer(
							(void **)&graph_slices[0]->d_column_indices,
							(void *) h_column_indices, 0),
						"CsrProblem cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror(cudaHostGetDevicePointer(
							(void **)&graph_slices[0]->d_row_offsets,
							(void *) h_row_offsets, 0),
						"CsrProblem cudaHostGetDevicePointer d_row_offsets failed", __FILE__, __LINE__)) break;

				} else {

					// Allocate and initialize d_column_indices

					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &graph_slices[0]->d_column_indices,
							graph_slices[0]->edges * sizeof(VertexId)),
						"CsrProblem cudaMalloc d_column_indices failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror(cudaMemcpy(
							graph_slices[0]->d_column_indices,
							h_column_indices,
							graph_slices[0]->edges * sizeof(VertexId),
							cudaMemcpyHostToDevice),
						"CsrProblem cudaMemcpy d_column_indices failed", __FILE__, __LINE__)) break;

					// Allocate and initialize d_row_offsets

					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &graph_slices[0]->d_row_offsets,
							(graph_slices[0]->nodes + 1) * sizeof(SizeT)),
						"CsrProblem cudaMalloc d_row_offsets failed", __FILE__, __LINE__)) break;

					if (retval = util::B40CPerror(cudaMemcpy(
							graph_slices[0]->d_row_offsets,
							h_row_offsets,
							(graph_slices[0]->nodes + 1) * sizeof(SizeT),
							cudaMemcpyHostToDevice),
						"CsrProblem cudaMemcpy d_row_offsets failed", __FILE__, __LINE__)) break;
				}

			} else {

				// Create multiple GPU graph slices
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					// Set device
					if (retval = util::B40CPerror(cudaSetDevice(gpu),
						"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

					// Create stream
					cudaStream_t stream;
					if (retval = util::B40CPerror(cudaStreamCreate(&stream),
						"CsrProblem cudaStreamCreate failed", __FILE__, __LINE__)) break;

					// Create slice
					graph_slices.push_back(new GraphSlice(gpu, stream));
				}
				if (retval) break;

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

					// Mask in owning gpu
					for (int gpu = 0; gpu < row_edges; gpu++) {
						VertexId *ptr = slice_column_indices[gpu] + slice_row_offsets[gpu][slice_row] + gpu;
						VertexId owner = GpuIndex(*ptr);
						(*ptr) |= (owner << ProblemType::GPU_MASK_SHIFT);
					}
				}

				printf("Done constructing gpu data structures\n");
				fflush(stdout);

				// Initialize data structures on GPU
				for (int gpu = 0; gpu < num_gpus; gpu++) {

					// Set device
					if (util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu),
						"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;

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
				}
				if (retval) break;

				if (slice_row_offsets) delete slice_row_offsets;
				if (slice_column_indices) delete slice_column_indices;
			}

		} while (0);

		return retval;
	}


	/**
	 * Performs any initialization work needed for this problem type.  Must be called
	 * prior to each search
	 */
	cudaError_t Reset(
		FrontierType frontier_type,						// The frontier type (i.e., edge/vertex/mixed)
		double queue_sizing = DEFAULT_QUEUE_SIZING)		// The scaling factor for frontier queue size (e.g., 1.3x edge-count) to accommodate redundant expansion
	{
		cudaError_t retval = cudaSuccess;

		for (int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu),
				"CsrProblem cudaSetDevice failed", __FILE__, __LINE__)) break;


			//
			// Allocate results output if necessary
			//

			if (!graph_slices[gpu]->d_labels) {

				printf("GPU %d source path: %lld elements (%lld bytes)\n",
					graph_slices[gpu]->gpu,
					(unsigned long long) graph_slices[gpu]->nodes,
					(unsigned long long) graph_slices[gpu]->nodes * sizeof(VertexId));

				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_labels,
						graph_slices[gpu]->nodes * sizeof(VertexId)),
					"CsrProblem cudaMalloc d_labels failed", __FILE__, __LINE__)) break;
			}


			//
			// Allocate visited masks for the entire graph if necessary
			//

			int visited_mask_bytes 			= ((nodes * sizeof(VisitedMask)) + 8 - 1) / 8;					// round up to the nearest VisitedMask
			int visited_mask_elements		= visited_mask_bytes * sizeof(VisitedMask);
			if (!graph_slices[gpu]->d_visited_mask) {

				printf("GPU %d visited mask: %lld elements (%lld bytes)\n",
					graph_slices[gpu]->gpu,
					(unsigned long long) visited_mask_elements,
					(unsigned long long) visited_mask_bytes);

				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->d_visited_mask,
						visited_mask_bytes),
					"CsrProblem cudaMalloc d_visited_mask failed", __FILE__, __LINE__)) break;
			}


			//
			// Allocate frontier queues if necessary
			//

			// Determine queue sizes
			SizeT new_ping_elements = double(graph_slices[gpu]->nodes) * queue_sizing;
			SizeT new_pong_elements = double(graph_slices[gpu]->edges) * queue_sizing;

			switch (frontier_type) {
			case VERTEX_FRONTIERS:
				new_pong_elements = new_ping_elements;
				break;
			case EDGE_FRONTIERS:
				new_ping_elements = new_pong_elements;
				break;
			};

			// Free queues if not big enough
			if (new_ping_elements > graph_slices[gpu]->ping_elements) {
				if (graph_slices[gpu]->frontier_queues.d_keys[0]) util::B40CPerror(cudaFree(graph_slices[gpu]->frontier_queues.d_keys[0]), "GpuSlice cudaFree frontier_queues.d_keys[0] failed", __FILE__, __LINE__);
				if (graph_slices[gpu]->frontier_queues.d_values[0]) util::B40CPerror(cudaFree(graph_slices[gpu]->frontier_queues.d_values[0]), "GpuSlice cudaFree frontier_queues.d_values[0] failed", __FILE__, __LINE__);
				graph_slices[gpu]->frontier_queues.d_keys[0] = NULL;
				graph_slices[gpu]->frontier_queues.d_values[0] = NULL;
				graph_slices[gpu]->ping_elements = new_ping_elements;
			}
			if (new_pong_elements > graph_slices[gpu]->pong_elements) {
				if (graph_slices[gpu]->frontier_queues.d_keys[1]) util::B40CPerror(cudaFree(graph_slices[gpu]->frontier_queues.d_keys[1]), "GpuSlice cudaFree frontier_queues.d_keys[1] failed", __FILE__, __LINE__);
				if (graph_slices[gpu]->frontier_queues.d_values[1]) util::B40CPerror(cudaFree(graph_slices[gpu]->frontier_queues.d_values[1]), "GpuSlice cudaFree frontier_queues.d_values[1] failed", __FILE__, __LINE__);
				graph_slices[gpu]->frontier_queues.d_keys[1] = NULL;
				graph_slices[gpu]->frontier_queues.d_values[1] = NULL;
				graph_slices[gpu]->pong_elements = new_pong_elements;
			}

			// Allocate ping if necessary
			if (!graph_slices[gpu]->frontier_queues.d_keys[0]) {

				printf("GPU %d frontier queue (ping): %lld elements (%lld bytes)\n",
					graph_slices[gpu]->gpu,
					(unsigned long long) graph_slices[gpu]->ping_elements,
					(unsigned long long) graph_slices[gpu]->ping_elements * sizeof(VertexId));
				fflush(stdout);

				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->frontier_queues.d_keys[0],
						graph_slices[gpu]->vertex_frontier_elements * sizeof(VertexId)),
					"CsrProblem cudaMalloc frontier_queues.d_keys[0] failed", __FILE__, __LINE__)) break;
			}

			// Allocate pong if necessary
			if (!graph_slices[gpu]->frontier_queues.d_keys[1]) {

				printf("GPU %d frontier queue (pong): %lld elements (%lld bytes)\n",
					graph_slices[gpu]->gpu,
					(unsigned long long) graph_slices[gpu]->pong_elements,
					(unsigned long long) graph_slices[gpu]->pong_elements * sizeof(VertexId));
				fflush(stdout);

				if (retval = util::B40CPerror(cudaMalloc(
						(void**) &graph_slices[gpu]->frontier_queues.d_keys[1],
						graph_slices[gpu]->edge_frontier_elements * sizeof(VertexId)),
					"CsrProblem cudaMalloc frontier_queues.d_keys[1] failed", __FILE__, __LINE__)) break;
			}

			// Allocate predecessor vertex queues if necessary
			if (MARK_PREDECESSORS) {
				// Ping
				if (!graph_slices[gpu]->frontier_queues.d_values[0]) {
					printf("GPU %d predecessor queue (ping): %lld elements (%lld bytes)\n",
						graph_slices[gpu]->gpu,
						(unsigned long long) graph_slices[gpu]->ping_elements,
						(unsigned long long) graph_slices[gpu]->ping_elements * sizeof(VertexId));
					fflush(stdout);

					if (retval = util::B40CPerror(
							cudaMalloc((void**) &graph_slices[gpu]->frontier_queues.d_values[0],
							graph_slices[gpu]->vertex_frontier_elements * sizeof(VertexId)),
						"CsrProblem cudaMalloc frontier_queues.d_values[0] failed", __FILE__, __LINE__)) break;
				}
				// Pong
				if (!graph_slices[gpu]->frontier_queues.d_values[1]) {
					printf("GPU %d predecessor queue (pong): %lld elements (%lld bytes)\n",
						graph_slices[gpu]->gpu,
						(unsigned long long) graph_slices[gpu]->pong_elements,
						(unsigned long long) graph_slices[gpu]->pong_elements * sizeof(VertexId));
					fflush(stdout);

					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &graph_slices[gpu]->frontier_queues.d_values[1],
							graph_slices[gpu]->edge_frontier_elements * sizeof(VertexId)),
						"CsrProblem cudaMalloc frontier_queues.d_values[1] failed", __FILE__, __LINE__)) break;
				}
			}



			//
			// Allocate multi-gpu structures
			//

			if (num_gpus > 1) {

				// Allocate d_keep if necessary
				if (!graph_slices[gpu]->d_keep) {

					printf("GPU %d_keep flags: %lld elements (%lld bytes)\n",
						graph_slices[gpu]->gpu,
						(unsigned long long) graph_slices[gpu]->edge_frontier_elements,
						(unsigned long long) graph_slices[gpu]->edge_frontier_elements * sizeof(ValidFlag));

					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &graph_slices[gpu]->d_keep,
							graph_slices[gpu]->edge_frontier_elements * sizeof(ValidFlag)),
						"CsrProblem cudaMalloc d_keep failed", __FILE__, __LINE__)) break;
				}

				// Allocate d_multigpu_vqueue if necessary
				if (!graph_slices[gpu]->d_multigpu_vqueue) {

					graph_slices[gpu]->multigpu_vqueue_elements = graph_slices[gpu]->nodes * 2;

					printf("GPU %d_multigpu_vqueue: %lld elements (%lld bytes)\n",
						graph_slices[gpu]->gpu,
						(unsigned long long) graph_slices[gpu]->multigpu_vqueue_elements,
						(unsigned long long) graph_slices[gpu]->multigpu_vqueue_elements * sizeof(VertexId));

					if (retval = util::B40CPerror(cudaMalloc(
							(void**) &graph_slices[gpu]->d_multigpu_vqueue,
							graph_slices[gpu]->multigpu_vqueue_elements * sizeof(VertexId)),
						"CsrProblem cudaMalloc d_multigpu_vqueue failed", __FILE__, __LINE__)) break;

				}
			}
			printf("\n");

			//
			// Initialize source paths and collision mask cache
			//

			int memset_block_size 		= 256;
			int memset_grid_size_max 	= 32 * 1024;	// 32K CTAs
			int memset_grid_size;

			// Initialize d_labels elements to -1
			memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[gpu]->nodes + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VertexId><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
				graph_slices[gpu]->d_labels,
				-1,
				graph_slices[gpu]->nodes);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

			// Initialize d_visited_mask elements to 0
			memset_grid_size = B40C_MIN(memset_grid_size_max, (visited_mask_elements + memset_block_size - 1) / memset_block_size);
			util::MemsetKernel<VisitedMask><<<memset_grid_size, memset_block_size, 0, graph_slices[gpu]->stream>>>(
				graph_slices[gpu]->d_visited_mask,
				0,
				visited_mask_elements);

			if (retval = util::B40CPerror(cudaThreadSynchronize(),
				"MemsetKernel failed", __FILE__, __LINE__)) break;

		}

		return retval;
	}
};


// Whether to mark predecessors vs mark distance-from-source
template <typename VertexId, typename SizeT, bool MARK_PREDECESSORS>
const float CsrProblem<VertexId, SizeT, MARK_PREDECESSORS>::DEFAULT_QUEUE_SIZING = 1.30;


} // namespace bfs
} // namespace graph
} // namespace b40c
