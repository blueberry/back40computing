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

#include <b40c/util/spine.cuh>
#include <b40c/util/cta_work_distribution.cuh>

#include <b40c/bfs/compact/problem_type.cuh>
#include <b40c/bfs/compact/problem_config.cuh>
#include <b40c/bfs/compact/upsweep_kernel_config.cuh>
#include <b40c/bfs/compact/upsweep_kernel.cuh>
#include <b40c/bfs/compact/downsweep_kernel_config.cuh>
#include <b40c/bfs/compact/downsweep_kernel.cuh>

#include <b40c/scan/spine_kernel.cuh>

namespace b40c {
namespace bfs {


/**
 * Level-grid breadth-first-search enactor.
 *  
 * Each iterations is performed by its own kernel-launch.  
 */
template <typename VertexId>
class LevelGridBfsEnactor : public BaseBfsEnactor<VertexId, BfsCsrProblem<VertexId> >
{
private:
	
	typedef BaseBfsEnactor<VertexId, BfsCsrProblem<VertexId> > Base;
	

protected:	

	/**
	 * Rotating 4-element array of atomic counters indicating sizes of the 
	 * incoming and outgoing frontier queues, where 
	 * incoming = iteration % 4, and outgoing = (iteration + 1) % 4
	 */
	int *d_queue_lengths;
	
	/**
	 * Temporary device storage needed for reducing partials produced
	 * by separate CTAs
	 */
	util::Spine spine;

	// Flag vector
	unsigned char *d_keep;

protected:
	
	/**
	 * Utility function: Returns the default maximum number of threadblocks 
	 * this enactor class can launch.
	 */
	static int MaxGridSize(const util::CudaProperties &props, int max_grid_size = 0)
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
		const util::CudaProperties &props = util::CudaProperties()) :
			Base::BaseBfsEnactor(max_queue_size, MaxGridSize(props, max_grid_size), props),
			d_queue_lengths(NULL),
			d_keep(NULL)
	{
		// Size of 4-element rotating vector of queue lengths (and statistics)   
		const int QUEUE_LENGTHS_SIZE = 4;
		
		// Allocate 
		if (cudaMalloc((void**) &d_queue_lengths, sizeof(int) * QUEUE_LENGTHS_SIZE)) {
			printf("LevelGridBfsEnactor:: cudaMalloc d_queue_lengths failed: ", __FILE__, __LINE__);
			exit(1);
		}
		if (cudaMalloc((void**) &d_keep, max_queue_size * sizeof(unsigned char))) {
			printf("LevelGridBfsEnactor:: cudaMalloc d_keep failed: ", __FILE__, __LINE__);
			exit(1);
		}

		// Initialize 
		util::MemsetKernel<int><<<1, QUEUE_LENGTHS_SIZE>>>(								// to zero
			d_queue_lengths, 0, QUEUE_LENGTHS_SIZE);
	}

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
		BfsCsrProblem<VertexId> &bfs_problem,
		VertexId src,
		BfsStrategy strategy) 
	{
		typedef int SizeT;

		const int CTA_THREADS = 1 << B40C_BFS_SG_LOG_CTA_THREADS(this->cuda_props.device_sm_version, strategy);

		cudaError_t retval = cudaSuccess;

		if (this->BFS_DEBUG) {
			printf("\n[BFS level-sync, %s config:] device_sm_version: %d, kernel_ptx_version: %d, grid_size: %d, threads: %d, max_queue_size: %d\n", 
				(strategy == CONTRACT_EXPAND) ? "contract-expand" : "expand-contract", 
				this->cuda_props.device_sm_version, this->cuda_props.kernel_ptx_version, 
				this->max_grid_size, CTA_THREADS, this->max_queue_size);
			fflush(stdout);
		}

		// Make sure our spine is big enough
		int spine_elements = this->max_grid_size;
		if (retval = spine.Setup<SizeT>(this->max_grid_size, spine_elements)) exit(1);

		// Set up compaction configuration types
		typedef bfs::compact::ProblemType<VertexId, SizeT> ProblemType;
		typedef bfs::compact::ProblemConfig<
			ProblemType,
			200,
			util::io::ld::NONE,
			util::io::st::NONE,
			9,

			8,
			7,
			0,
			1,

			5,
			2,
			0,
			5,

			8,
			7,
			2,
			0,
			5> ProblemConfig;

		typedef typename ProblemConfig::Upsweep Upsweep;
		typedef typename ProblemConfig::Spine Spine;
		typedef typename ProblemConfig::Downsweep Downsweep;




		int queue_idx 			= 0;
		int num_elements 		= 1;
		VertexId iteration 		= 0;

		while (true) {

			// Contract-expand strategy
			BfsLevelGridKernel<VertexId, CONTRACT_EXPAND><<<this->max_grid_size, CTA_THREADS>>>(
				src,
				bfs_problem.d_collision_cache,
				this->d_queue[queue_idx],
				this->d_queue[queue_idx ^ 1],
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_dist,
				this->d_queue_lengths,
				iteration);

			if (BFS_DEBUG && cudaThreadSynchronize()) {
				printf("BfsLevelGridKernel failed: %d %d", __FILE__, __LINE__);
				exit(1);
			}

			// Update out-queue length
			int outgoing_queue_length_idx = (iteration + 1) & 0x3;
			if (cudaMemcpy(
				&num_elements,
				d_queue_lengths + outgoing_queue_length_idx,
				1 * sizeof(int),
				cudaMemcpyDeviceToHost))
			{
				printf("cudaMemcpy failed: %d %d", __FILE__, __LINE__);
				exit(1);
			}

			printf("Iteration %d output queued %d nodes\n", iteration, num_elements);

			if (num_elements == 0) {
				// No more work, all done.
				break;
			}

			queue_idx ^= 1;

			// Obtain a CTA work distribution for copying items of type T
			util::CtaWorkDistribution<SizeT> work(
				num_elements,
				Upsweep::SCHEDULE_GRANULARITY,
				this->max_grid_size);

			// Upsweep
			bfs::compact::UpsweepKernel<Upsweep><<<this->max_grid_size, Upsweep::THREADS>>>(
				this->d_queue[queue_idx],
				this->d_keep,
				(SizeT *) this->spine(),
				bfs_problem.d_collision_cache,
				work);

			// Spine
			scan::SpineKernel<Spine><<<1, Spine::THREADS>>>(
				(SizeT*) spine(), (SizeT*) spine(), spine_elements);

			// Downsweep
			bfs::compact::DownsweepKernel<Downsweep><<<this->max_grid_size, Downsweep::THREADS>>>(
				this->d_queue[queue_idx],
				this->d_keep,
				this->d_queue_lengths + outgoing_queue_length_idx,
				this->d_queue[queue_idx ^ 1],
				(SizeT *) this->spine(),
				work);
/*
			// Update in-queue length
			if (cudaMemcpy(
				&num_elements,
				d_queue_lengths + outgoing_queue_length_idx,
				1 * sizeof(int),
				cudaMemcpyDeviceToHost))
			{
				printf("cudaMemcpy failed: %d %d", __FILE__, __LINE__);
				exit(1);
			}

			printf("Reduced to %d nodes\n\n", num_elements);

			if (num_elements == 0) {
				// No more work, all done.
				break;
			}
*/
			queue_idx ^= 1;

			iteration++;
		}

		return retval;
	}
    
};




} // namespace bfs
} // namespace b40c

