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

#include <bfs_base.cu>
#include <bfs_level_grid_kernel.cu>

#include <radixsort_api_enactor_tuned.cuh>		// Sorting includes
#include <radixsort_api_storage.cuh>			// Sorting includes

#include <b40c/consecutive_removal_enactor.cuh>

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
//	LsbSortEnactorTuned sorting_enactor;
//	ConsecutiveRemovalEnactor removal_enactor;

protected:	

	/**
	 * Rotating 4-element array of atomic counters indicating sizes of the 
	 * incoming and outgoing frontier queues, where 
	 * incoming = iteration % 4, and outgoing = (iteration + 1) % 4
	 */
	int *d_queue_lengths;
	
	char *d_keep;

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
		cudaMalloc((void**) &d_keep, sizeof(char) * this->max_queue_size);
		
		// Initialize 
		MemsetKernel<int><<<1, QUEUE_LENGTHS_SIZE>>>(								// to zero
			d_queue_lengths, 0, QUEUE_LENGTHS_SIZE);
		
		// Setup cache config for kernels
//		cudaFuncSetCacheConfig(BfsLevelKernel<IndexType, CONTRACT_EXPAND>, cudaFuncCachePreferL1);
//		cudaFuncSetCacheConfig(BfsLevelKernel<IndexType, EXPAND_CONTRACT>, cudaFuncCachePreferL1);
	}

public:

	/**
	 * Verify the contents of a device array match those
	 * of a host array
	 */
	template <typename T>
	void DisplayDeviceResults(
		T *d_data,
		size_t num_elements)
	{
		// Allocate array on host
		T *h_data = (T*) malloc(num_elements * sizeof(T));

		// Reduction data back
		cudaMemcpy(h_data, d_data, sizeof(T) * num_elements, cudaMemcpyDeviceToHost);

		// Display data
		printf("\n\nData:\n");
		for (int i = 0; i < num_elements; i++) {
			PrintValue(h_data[i]);
			printf(", ");
		}
		printf("\n\n");

		// Cleanup
		if (h_data) free(h_data);
	}

	/**
     * Destructor
     */
    virtual ~LevelGridBfsEnactor() 
    {
		if (d_queue_lengths) cudaFree(d_queue_lengths);
		if (d_keep) cudaFree(d_keep);
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
		BfsCsrProblem<IndexType> &bfs_problem,
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
		MultiCtaSortStorage<unsigned int> sorting_problem;
		sorting_problem.d_keys[0] = (unsigned int *) this->d_queue[0];
		sorting_problem.d_keys[1] = (unsigned int *) this->d_queue[1];
		sorting_problem.selector = 0;
		sorting_problem.num_elements = 1;

		int iteration = 0;

		while (true) {
/*
			MemsetKernel<char><<<128, 128>>>(d_keep, 1, max_queue_size * sizeof(char));

			// Contract-expand strategy
			BfsLevelGridKernel<IndexType, CULL><<<this->max_grid_size, cta_threads>>>(
				src,
				bfs_problem.d_collision_cache,
				d_keep,
				this->d_queue[sorting_problem.selector],
				this->d_queue[sorting_problem.selector ^ 1],
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_dist,
				this->d_queue_lengths,
				iteration);
			dbg_sync_perror_exit("LevelGridRadixSortingEnactor:: BfsLevelGridKernel failed: ", __FILE__, __LINE__);

			// mooch
			char *keep = (char *) malloc(sorting_problem.num_elements);
			cudaMemcpy(keep, d_keep, sizeof(char) * sorting_problem.num_elements, cudaMemcpyDeviceToHost);
			int sum = 0;
			for (int i = 0; i < sorting_problem.num_elements; i++) {
				sum += keep[i];
			}
			printf("    Iteration %d input reduced to %d\n", iteration, sum);
			free(keep);
*/
			// Contract-expand strategy
			BfsLevelGridKernel<IndexType, CONTRACT_EXPAND><<<this->max_grid_size, cta_threads>>>(
				src,
				bfs_problem.d_collision_cache,
				NULL,
				this->d_queue[sorting_problem.selector],
				this->d_queue[sorting_problem.selector ^ 1],
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_dist,
				this->d_queue_lengths,
				iteration);
			dbg_sync_perror_exit("LevelGridRadixSortingEnactor:: BfsLevelGridKernel failed: ", __FILE__, __LINE__);

			// Update out-queue length
			int outgoing_queue_length_idx = (iteration + 1) & 0x3;
			cudaMemcpy(&sorting_problem.num_elements, d_queue_lengths + outgoing_queue_length_idx,
				1 * sizeof(int), cudaMemcpyDeviceToHost);

			if (sorting_problem.num_elements == 0) {
				// No more work, all done.
				break;
			}
			printf("Iteration %d output queued %d nodes\n", iteration, sorting_problem.num_elements);

			// Update incoming-queue selector
			sorting_problem.selector ^= 1;

/*
			// Sort outgoing queue
			sorting_problem.num_elements = sorting_problem.num_elements;
			sorting_enactor.EnactSort(sorting_problem);

			removal_enactor.Enact(
				sorting_problem.d_keys[sorting_problem.selector ^ 1],
				d_queue_lengths + outgoing_queue_length_idx,
				sorting_problem.d_keys[sorting_problem.selector],
				sorting_problem.num_elements);

			// Update incoming-queue selector
			sorting_problem.selector ^= 1;

			cudaMemcpy(&sorting_problem.num_elements, d_queue_lengths + outgoing_queue_length_idx,
				1 * sizeof(int), cudaMemcpyDeviceToHost);

			printf("Iteration %d reduced to %d nodes\n", iteration, sorting_problem.num_elements);
*/
			iteration++;
		}

		return cudaSuccess;
	}
    
};





}// namespace b40c

