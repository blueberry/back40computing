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
 * API a the Single-grid BFS Imlementation
 ******************************************************************************/

#pragma once

#include <b40c_error_synchronize.cu>

#include <bfs_base.cu>
#include <bfs_level_grid_kernel.cu>

#include <radixsort_early_exit.cu>		// Sorting includes

namespace b40c {

/**
 * Level-grid breadth-first-search enactor.
 *  
 * Each iterations is performed by its own kernel-launch.  
 */
template <typename IndexType>
class LevelGridBfsEnactor : public BaseBfsEnactor<IndexType, BfsCsrProblem<IndexType> >
{
private:
	
	typedef BaseBfsEnactor<IndexType, BfsCsrProblem<IndexType> > Base; 
	
	// Sorting enactor
	EarlyExitRadixSortingEnactor<IndexType> sorting_enactor;

protected:	

	/**
	 * Rotating 4-element array of atomic counters indicating sizes of the 
	 * incoming and outgoing frontier queues, where 
	 * incoming = iteration % 4, and outgoing = (iteration + 1) % 4
	 */
	int *d_queue_lengths;
	
protected:
	
	/**
	 * Utility function: Returns the default maximum number of threadblocks 
	 * this enactor class can launch.
	 */
	static int MaxGridSize(const CudaProperties &props, int max_grid_size = 0) 
	{
		if (max_grid_size == 0) {
			// No override: Fully populate all SMs
			max_grid_size = props.device_props.multiProcessorCount * 
					B40C_BFS_SG_OCCUPANCY(props.kernel_ptx_version); 
		} 
		
		return max_grid_size;
	}
	
public: 	
	
	/**
	 * Constructor.  Requires specification of the maximum number of elements
	 * that can be queued into the frontier-queue for a given BFS iteration.
	 */
	LevelGridBfsEnactor(
		int max_queue_size,
		int max_grid_size = 0,
		const CudaProperties &props = CudaProperties()) :
			Base::BaseBfsEnactor(max_queue_size, MaxGridSize(props, max_grid_size), props)
	{
		// Size of 4-element rotating vector of queue lengths (and statistics)   
		const int QUEUE_LENGTHS_SIZE = 4;
		
		// Allocate 
		cudaMalloc((void**) &d_queue_lengths, sizeof(int) * QUEUE_LENGTHS_SIZE);
		
		// Initialize 
		MemsetKernel<int><<<1, QUEUE_LENGTHS_SIZE>>>(								// to zero
			d_queue_lengths, 0, QUEUE_LENGTHS_SIZE);
		
		// Setup cache config for kernels
//		cudaFuncSetCacheConfig(BfsLevelKernel<IndexType, CONTRACT_EXPAND>, cudaFuncCachePreferL1);
//		cudaFuncSetCacheConfig(BfsLevelKernel<IndexType, EXPAND_CONTRACT>, cudaFuncCachePreferL1);
	}

public:

	/**
     * Destructor
     */
    virtual ~LevelGridBfsEnactor() 
    {
		if (d_queue_lengths) cudaFree(d_queue_lengths);
    }
    
    /**
     * Obtain statistics about the last BFS search enacted 
     */
    void GetStatistics(
    	int &total_queued, 
    	int &passes, 
    	double &avg_barrier_wait)		// total time spent waiting in barriers in ms (threadblock average)
    {
    	total_queued = 0;
    	passes = 0;
    	avg_barrier_wait = 0;
    }
    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSearch(
		BfsCsrProblem<IndexType> &problem_storage, 
		IndexType src, 
		BfsStrategy strategy) 
	{
		int cta_threads = 1 << B40C_BFS_SG_LOG_CTA_THREADS(this->cuda_props.device_sm_version, strategy);
		
		if (this->BFS_DEBUG) {
			printf("\n[BFS level-sync, %s config:] device_sm_version: %d, kernel_ptx_version: %d, grid_size: %d, threads: %d, max_queue_size: %d\n", 
				(strategy == CONTRACT_EXPAND) ? "contract-expand" : "expand-contract", 
				this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version, 
				this->max_grid_size, cta_threads, this->max_queue_size);
			fflush(stdout);
		}


		// Create 
		MultiCtaRadixSortStorage<IndexType> sorting_problem;
		sorting_problem.d_keys[0] = this->d_queue[0];
		sorting_problem.d_keys[1] = this->d_queue[1];
		sorting_problem.selector = 0;
		
		int iteration = 0;
		while (true) {
		
			switch (strategy) {
	
			case CONTRACT_EXPAND:
	
				// Contract-expand strategy
				BfsLevelGridKernel<IndexType, CONTRACT_EXPAND><<<this->max_grid_size, cta_threads>>>(
					src,
					this->d_queue[sorting_problem.selector],								
					this->d_queue[sorting_problem.selector ^ 1],								
					problem_storage.d_column_indices,				
					problem_storage.d_row_offsets,					 
					problem_storage.d_source_dist,					
					this->d_queue_lengths,							
					iteration);
				dbg_sync_perror_exit("LevelGridRadixSortingEnactor:: BfsLevelGridKernel failed: ", __FILE__, __LINE__);
	
				break;
				
			case EXPAND_CONTRACT:
				
				// Expand-contract strategy
				BfsLevelGridKernel<IndexType, EXPAND_CONTRACT><<<this->max_grid_size, cta_threads>>>(
					src,
					this->d_queue[sorting_problem.selector],								
					this->d_queue[sorting_problem.selector ^ 1],								
					problem_storage.d_column_indices,				
					problem_storage.d_row_offsets,					 
					problem_storage.d_source_dist,					
					this->d_queue_lengths,
					iteration);
				dbg_sync_perror_exit("LevelGridRadixSortingEnactor:: BfsLevelGridKernel failed: ", __FILE__, __LINE__);
				
				break;
	
			}
			
			// Retrieve out-queue length
			int outgoing_queue_length;
			int outgoing_queue_length_idx = (iteration + 1) & 0x3; 				
			cudaMemcpy(&outgoing_queue_length, d_queue_lengths + outgoing_queue_length_idx, 
				1 * sizeof(int), cudaMemcpyDeviceToHost);
			
			if (outgoing_queue_length == 0) {
				// No more work, all done.
				break;
			}

			printf("Iteration %d queued %d nodes\n", iteration, outgoing_queue_length);

			// Update incoming-queue selector
			sorting_problem.selector ^= 1;
			
			// Sort outgoing queue
			sorting_problem.num_elements = outgoing_queue_length;
			sorting_enactor.EnactSort(sorting_problem);
			
			iteration++;
		}
		
		return cudaSuccess;
	}
    
};





}// namespace b40c

