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
 * API for BFS Implementations
 ******************************************************************************/

#pragma once

#include <bfs_kernel_common.cu>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>

namespace b40c {
namespace bfs {

/******************************************************************************
 * ProblemStorage management structures for BFS problem data 
 ******************************************************************************/


/**
 * CSR storage management structure for BFS problems.  
 * 
 * It is the caller's responsibility to free any non-NULL storage arrays when
 * no longer needed.  This allows for the storage to be re-used for subsequent 
 * BFS operations on the same graph.
 */
template <typename VertexId>
struct BfsCsrProblem
{
	VertexId nodes;
	VertexId *d_column_indices;
	VertexId *d_row_offsets;
	VertexId *d_source_dist;
	unsigned char *d_collision_cache;
	
	BfsCsrProblem() : nodes(0), d_column_indices(NULL), d_row_offsets(NULL), d_source_dist(NULL) {}
 
	BfsCsrProblem(
		VertexId nodes,
		VertexId *d_column_indices,
		VertexId *d_row_offsets,
		VertexId *d_source_dist,
		unsigned char *d_collision_cache) :
			nodes(nodes), 
			d_column_indices(d_column_indices), 
			d_row_offsets(d_row_offsets), 
			d_source_dist(d_source_dist),
			d_collision_cache(d_collision_cache) {}
	
	BfsCsrProblem(
		VertexId nodes,
		VertexId edges,
		VertexId *h_column_indices,
		VertexId *h_row_offsets): nodes(nodes)
	{
		if (cudaMalloc((void**) &d_column_indices, edges * sizeof(VertexId))) {
			printf("BfsCsrProblem:: cudaMalloc d_column_indices failed: ", __FILE__, __LINE__);
			exit(1);
		}
		if (cudaMalloc((void**) &d_row_offsets, (nodes + 1) * sizeof(VertexId))) {
			printf("BfsCsrProblem:: cudaMalloc d_row_offsets failed: ", __FILE__, __LINE__);
			exit(1);
		}
		if (cudaMalloc((void**) &d_source_dist, nodes * sizeof(VertexId))) {
			printf("BfsCsrProblem:: cudaMalloc d_source_dist failed: ", __FILE__, __LINE__);
			exit(1);
		}

		unsigned int bitmask_bytes = ((nodes * sizeof(unsigned char)) + 8 - 1) / 8;
		if (cudaMalloc((void**) &d_collision_cache, bitmask_bytes)) {
			printf("BfsCsrProblem:: cudaMalloc d_collision_cache failed: ", __FILE__, __LINE__);
			exit(1);
		}

		cudaMemcpy(d_column_indices, h_column_indices, edges * sizeof(VertexId), cudaMemcpyHostToDevice);
		cudaMemcpy(d_row_offsets, h_row_offsets, (nodes + 1) * sizeof(VertexId), cudaMemcpyHostToDevice);
		
		ResetSourceDist();
	}

	/**
	 * Initialize d_source_dist to -1
	 */
	void ResetSourceDist() 
	{
		int max_grid_size = B40C_MIN(1024 * 32, (nodes + 128 - 1) / 128);
		util::MemsetKernel<VertexId><<<max_grid_size, 128>>>(d_source_dist, -1, nodes);

		unsigned int bitmask_bytes = ((nodes * sizeof(unsigned char)) + 8 - 1) / 8;
		util::MemsetKernel<unsigned char><<<128, 128>>>(d_collision_cache, 0, bitmask_bytes);
	}
	
	void Free() 
	{
		if (d_column_indices) { cudaFree(d_column_indices); d_column_indices = NULL; }
		if (d_row_offsets) { cudaFree(d_row_offsets); d_row_offsets = NULL; }
		if (d_source_dist) { cudaFree(d_source_dist); d_source_dist = NULL; }
		if (d_collision_cache) { cudaFree(d_collision_cache); d_collision_cache = NULL; }
	}
};



/**
 * Base class for breadth-first-search enactors.
 * 
 * A BFS search iteratively expands outwards from the given source node.  At 
 * each iteration, the algorithm discovers unvisited nodes that are adjacent 
 * to the nodes discovered by the previous iteration.  The first iteration 
 * discovers the source node. 
 */
template <typename VertexId, typename ProblemStorage>
class BaseBfsEnactor 
{
protected:	

	//Device properties
	const util::CudaProperties cuda_props;
	
	// Max grid size we will ever launch
	int max_grid_size;

	// Elements to allocate for a frontier queue
	int max_queue_size;
	
	// Frontier queues
	VertexId *d_queue[2];
	
public:

	// Allows display to stdout of search details
	bool BFS_DEBUG;

protected: 	
	
	/**
	 * Constructor.
	 */
	BaseBfsEnactor(
		int max_queue_size,
		int max_grid_size,
		const util::CudaProperties &props = util::CudaProperties()) :
			max_queue_size(max_queue_size), max_grid_size(max_grid_size), cuda_props(props), BFS_DEBUG(false)
	{
		if (cudaMalloc((void**) &d_queue[0], max_queue_size * sizeof(VertexId))) {
			printf("BaseBfsEnactor:: cudaMalloc d_queue[0] failed: ", __FILE__, __LINE__);
			exit(1);
		}
		if (cudaMalloc((void**) &d_queue[1], max_queue_size * sizeof(VertexId))) {
			printf("BaseBfsEnactor:: cudaMalloc d_queue[1] failed: ", __FILE__, __LINE__);
			exit(1);
		}
	}


public:

	/**
     * Destructor
     */
    virtual ~BaseBfsEnactor() 
    {
		if (d_queue[0]) cudaFree(d_queue[0]);
		if (d_queue[1]) cudaFree(d_queue[1]);
    }
    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSearch(
		ProblemStorage &problem_storage, 
		VertexId src,
		BfsStrategy strategy) = 0;	
    
};




} // namespace bfs
} // namespace b40c

