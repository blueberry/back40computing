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
#include <bfs_single_grid_kernel.cu>

namespace b40c {

/**
 * Single-grid breadth-first-search enactor.
 *  
 * All iterations are performed by a single kernel-launch.  This is 
 * made possible by software global-barriers across threadblocks.  
 * breadth-first-search enactors.
 */
template <typename IndexType, bool INSTRUMENTED>
class SingleGridBfsEnactor : public BaseBfsEnactor<IndexType, BfsCsrProblem<IndexType> >
{
private:
	
	typedef BaseBfsEnactor<IndexType, BfsCsrProblem<IndexType> > Base; 

protected:	

	/**
	 * Array of global synchronization counters, one for each threadblock
	 */
	int *d_sync;
	
	/**
	 * Rotating 4-element array of atomic counters indicating sizes of the 
	 * incoming and outgoing frontier queues, where 
	 * incoming = iteration % 4, and outgoing = (iteration + 1) % 4
	 */
	int *d_queue_lengths;
	
	/**
	 * Time (in clocks) spent by each threadblock in software global barrier.
	 * Can be used to measure load imbalance.
	 */
	unsigned long long *d_barrier_time;
	
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
	SingleGridBfsEnactor(
		int max_queue_size,
		int max_grid_size = 0,
		const CudaProperties &props = CudaProperties()) :
			Base::BaseBfsEnactor(max_queue_size, MaxGridSize(props, max_grid_size), props)
	{
		// Size of 4-element rotating vector of queue lengths (and statistics)   
		const int QUEUE_LENGTHS_SIZE = 4 + 2;
		
		// Allocate 
		cudaMalloc((void**) &d_sync, sizeof(int) * this->max_grid_size);
		cudaMalloc((void**) &d_queue_lengths, sizeof(int) * QUEUE_LENGTHS_SIZE);
		cudaMalloc((void**) &d_barrier_time, sizeof(unsigned long long) * this->max_grid_size);

		// Mooch
		synchronize("SingleGridBfsEnactor: post-malloc");
		
		// Initialize 
		MemsetKernel<int><<<(this->max_grid_size + 128 - 1) / 128, 128>>>(			// to zero
			d_sync, 0, this->max_grid_size);
		MemsetKernel<int><<<1, QUEUE_LENGTHS_SIZE>>>(								// to zero
			d_queue_lengths, 0, QUEUE_LENGTHS_SIZE);
		
		// Setup cache config for kernels
//		cudaFuncSetCacheConfig(BfsSingleGridKernel<IndexType, CONTRACT_EXPAND>, cudaFuncCachePreferL1);
//		cudaFuncSetCacheConfig(BfsSingleGridKernel<IndexType, EXPAND_CONTRACT>, cudaFuncCachePreferL1);
	}

public:

	/**
     * Destructor
     */
    virtual ~SingleGridBfsEnactor() 
    {
		if (d_sync) cudaFree(d_sync);
		if (d_queue_lengths) cudaFree(d_queue_lengths);
    }
    
    /**
     * Obtain statistics about the last BFS search enacted 
     */
    void GetStatistics(
    	int &total_queued, 
    	int &passes, 
    	unsigned long long &avg_barrier_wait)		// in cycles
    {
    	total_queued = 0;
    	passes = 0;
    	avg_barrier_wait = 0;
    	
    	if (INSTRUMENTED) {
    	
			cudaMemcpy(&total_queued, d_queue_lengths + 4, 1 * sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&passes, d_queue_lengths + 5, 1 * sizeof(int), cudaMemcpyDeviceToHost);
	
			unsigned long long *h_barrier_time = 
				(unsigned long long *) malloc(this->max_grid_size * sizeof(unsigned long long));
			
			cudaMemcpy(h_barrier_time, d_barrier_time, this->max_grid_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
			unsigned long long total_barrier_time = 0;
			for (int i = 0; i < this->max_grid_size; i++) {
				total_barrier_time += h_barrier_time[i] / passes;
			}
			
			avg_barrier_wait = total_barrier_time / this->max_grid_size;
			
			free(h_barrier_time);
    	}
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
		
		switch (strategy) {

		case CONTRACT_EXPAND:

			// Contract-expand strategy
			if (BFS_DEBUG) {
				printf("\n[BFS contract-expand config:] device_sm_version: %d, kernel_ptx_version: %d, grid_size: %d, threads: %d, max_queue_size: %d\n", 
					this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version, 
					this->max_grid_size, cta_threads, this->max_queue_size);
				fflush(stdout);
			}

    		BfsSingleGridKernel<IndexType, CONTRACT_EXPAND, INSTRUMENTED><<<this->max_grid_size, cta_threads>>>(
				src,
				this->d_in_queue,								
				this->d_out_queue,								
				problem_storage.d_column_indices,				
				problem_storage.d_row_offsets,					 
				problem_storage.d_source_dist,					
				this->d_queue_lengths,							
				this->d_sync,
				this->d_barrier_time);
				
			break;
    		
		case EXPAND_CONTRACT:
			
			// Expand-contract strategy
			if (BFS_DEBUG) {
				printf("\n[BFS expand-contract config:] device_sm_version: %d, kernel_ptx_version: %d, grid_size: %d, threads: %d, max_queue_size: %d\n", 
					this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version, 
					this->max_grid_size, cta_threads, this->max_queue_size);
				fflush(stdout);
			}
			
			BfsSingleGridKernel<IndexType, EXPAND_CONTRACT, INSTRUMENTED><<<this->max_grid_size, cta_threads>>>(
				src,
				this->d_in_queue,								
				this->d_out_queue,								
				problem_storage.d_column_indices,				
				problem_storage.d_row_offsets,					 
				problem_storage.d_source_dist,					
				this->d_queue_lengths,							
				this->d_sync,
				this->d_barrier_time);
			
			break;

		}
		
		return cudaSuccess;
	}
    
};





}// namespace b40c

