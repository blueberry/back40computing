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

#include <b40c_error_synchronize.cu>
#include <bfs_kernel_common.cu>

namespace b40c {


// Debugging options
static bool BFS_DEBUG = false;


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
template <typename IndexType> 
struct BfsCsrProblem
{
	IndexType nodes;
	IndexType *d_column_indices;
	IndexType *d_row_offsets;
	IndexType *d_source_dist;
	
	BfsCsrProblem() : nodes(0), d_column_indices(NULL), d_row_offsets(NULL), d_source_dist(NULL) {}
 
	BfsCsrProblem(
		IndexType nodes,
		IndexType *d_column_indices, 
		IndexType *d_row_offsets, 
		IndexType *d_source_dist) : 
			nodes(nodes), 
			d_column_indices(d_column_indices), 
			d_row_offsets(d_row_offsets), 
			d_source_dist(d_source_dist) {}
	
	BfsCsrProblem(
		IndexType nodes,
		IndexType edges,
		IndexType *h_column_indices, 
		IndexType *h_row_offsets): nodes(nodes)
	{
		cudaMalloc((void**) &d_column_indices, edges * sizeof(IndexType));
		cudaMalloc((void**) &d_row_offsets, (nodes + 1) * sizeof(IndexType));
		cudaMalloc((void**) &d_source_dist, nodes * sizeof(IndexType));

		// Mooch
		synchronize("BfsCsrProblem: post-malloc");
		
		cudaMemcpy(d_column_indices, h_column_indices, edges * sizeof(IndexType), cudaMemcpyHostToDevice);
		cudaMemcpy(d_row_offsets, h_row_offsets, (nodes + 1) * sizeof(IndexType), cudaMemcpyHostToDevice);
		
		ResetSourceDist();
	}

	/**
	 * Initialize d_source_dist to -1
	 */
	void ResetSourceDist() 
	{
		int max_grid_size = B40C_MIN(1024 * 32, (nodes + 128 - 1) / 128);
		MemsetKernel<int><<<max_grid_size, 128>>>(d_source_dist, -1, nodes);
	}
	
	void Free() 
	{
		if (d_column_indices) { cudaFree(d_column_indices); d_column_indices = NULL; }
		if (d_row_offsets) { cudaFree(d_row_offsets); d_row_offsets = NULL; }
		if (d_source_dist) { cudaFree(d_source_dist); d_source_dist = NULL; }
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
template <typename IndexType, typename ProblemStorage>
class BaseBfsEnactor 
{
protected:	

	//Device properties
	const CudaProperties cuda_props;
	
	// Max grid size we will ever launch
	int max_grid_size;

	// Elements to allocate for a frontier queue
	int max_queue_size;
	
	// Frontier queues
	IndexType *d_in_queue;
	IndexType *d_out_queue;
	
protected: 	
	
	/**
	 * Constructor.
	 */
	BaseBfsEnactor(
		int max_queue_size,
		int max_grid_size,
		const CudaProperties &props = CudaProperties()) : 
			max_queue_size(max_queue_size), max_grid_size(max_grid_size), cuda_props(props) 
	{
		cudaMalloc((void**) &d_in_queue, max_queue_size * sizeof(IndexType));
		cudaMalloc((void**) &d_out_queue, max_queue_size * sizeof(IndexType));
		
		// Mooch
		synchronize("BaseBfsEnactor: post-malloc");
	}

public:

	/**
     * Destructor
     */
    virtual ~BaseBfsEnactor() 
    {
		if (d_in_queue) cudaFree(d_in_queue);
		if (d_out_queue) cudaFree(d_out_queue);
    }
    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSearch(
		ProblemStorage &problem_storage, 
		IndexType src, 
		BfsStrategy strategy) = 0;	
    
};



}// namespace b40c

