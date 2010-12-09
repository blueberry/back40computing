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
template <typename IndexType>
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
		// Allocate and initialize synchronization counters to zero
		cudaMalloc((void**) &d_sync, sizeof(int) * this->max_grid_size);
		MemsetKernel<int><<<(this->max_grid_size + 128 - 1) / 128, 128>>>(
			d_sync, 0, this->max_grid_size);
		
		// Allocate and initialize 4-element rotating vector of queue lengths to zero
		cudaMalloc((void**) &d_queue_lengths, sizeof(int) * 4);
		MemsetKernel<int><<<1, 4>>>(d_queue_lengths, 0, 4);
		
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
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
	virtual cudaError_t EnactSearch(
		BfsCsrProblem<IndexType> &problem_storage, 
		IndexType src, 
		BfsStrategy strategy) 
	{
		switch (strategy) {
		case CONTRACT_EXPAND:
			
			if (BFS_DEBUG) {
				printf("\n[BFS contract-expand config:] device_sm_version: %d, kernel_ptx_version: %d, grid_size: %d, threads: %d, max_queue_size: %d\n", 
					this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version, this->max_grid_size, B40C_BFS_SG_THREADS, this->max_queue_size);
				fflush(stdout);
			}

    		BfsSingleGridKernel<IndexType, CONTRACT_EXPAND><<<this->max_grid_size, B40C_BFS_SG_THREADS>>>(
				src,
				this->d_in_queue,								
				this->d_out_queue,								
				problem_storage.d_column_indices,				
				problem_storage.d_row_offsets,					 
				problem_storage.d_source_dist,					
				this->d_queue_lengths,							
				this->d_sync);
				
			break;

/*    		
		case EXPAND_CONTRACT:
			
    		printf("\nDevice_sm_version: %d, kernel_ptx_version: %d\n", 
    				this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version);
    		printf("EXPAND_CONTRACT:\n\tgrid_size: %d, \n\tthreads: %d\n\n",
    				this->max_grid_size, B40C_BFS_SG_THREADS);
			
			BfsSingleGridKernel<IndexType, EXPAND_CONTRACT><<<this->max_grid_size, B40C_BFS_SG_THREADS>>>(
				src,
				this->d_in_queue,								
				this->d_out_queue,								
				problem_storage.d_column_indices,				
				problem_storage.d_row_offsets,					 
				problem_storage.d_source_dist,					
				this->d_queue_lengths,							
				this->d_sync);
			break;
*/			
		}
		
		return cudaSuccess;
	}
    
};





}// namespace b40c

