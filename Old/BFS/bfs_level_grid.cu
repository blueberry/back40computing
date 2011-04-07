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
#include <bfs_common.cu>

#include <b40c/util/spine.cuh>
#include <b40c/bfs/problem_type.cuh>
#include <b40c/bfs/problem_config.cuh>

#include <b40c/bfs/expand_atomic/sweep_kernel.cuh>
#include <b40c/bfs/compact/upsweep_kernel.cuh>
#include <b40c/bfs/compact/downsweep_kernel.cuh>
#include <b40c/scan/spine_kernel.cuh>

namespace b40c {
namespace bfs {


/**
 * Level-grid breadth-first-search enactor.
 *  
 * Each iterations is performed by its own kernel-launch.  
 */
class LevelGridBfsEnactor : public BaseBfsEnactor
{
protected:
	
	/**
	 * Temporary device storage needed for reducing partials produced
	 * by separate CTAs
	 */
	util::Spine spine;

protected:

	/**
	 * Utility function: Returns the default maximum number of threadblocks 
	 * this enactor class can launch.
	 */
	int MaxGridSize(int cta_occupancy, int max_grid_size = 0)
	{
		if (max_grid_size <= 0) {
			// No override: Fully populate all SMs
			max_grid_size = this->cuda_props.device_props.multiProcessorCount * cta_occupancy;
		} 
		
		return max_grid_size;
	}
	
public: 	
	
	/**
	 * Constructor
	 */
	LevelGridBfsEnactor(bool DEBUG = false) : BaseBfsEnactor(DEBUG) {}


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
    template <BfsStrategy STRATEGY, typename BfsCsrProblem>
	cudaError_t EnactSearch(
		BfsCsrProblem 						&bfs_problem,
		typename BfsCsrProblem::VertexId 	src,
		int 								max_grid_size = 0)
	{
		// Compaction tuning configuration
		typedef ProblemConfig<
			typename BfsCsrProblem::ProblemType,
			200,
			util::io::ld::NONE,
			util::io::st::NONE,
			9,

			// Atomic expand
			6,
			8,
			7,
			0,
			0,
			5,
			true,

			// Compact upsweep
			8,
			7,
			0,
			0,

			// Compact spine
			5,
			2,
			0,
			5,

			// Compact downsweep
			8,
			7,
			1,
			1,
			5> ProblemConfig;

		typedef typename ProblemConfig::ExpandAtomicSweep 	ExpandAtomicSweep;
		typedef typename ProblemConfig::CompactUpsweep 		CompactUpsweep;
		typedef typename ProblemConfig::CompactSpine 		CompactSpine;
		typedef typename ProblemConfig::CompactDownsweep 	CompactDownsweep;

		typedef typename BfsCsrProblem::VertexId			VertexId;
		typedef typename BfsCsrProblem::SizeT				SizeT;


		cudaError_t retval = cudaSuccess;

		// Determine grid size
		int min_occupancy = B40C_MIN(CompactUpsweep::CTA_OCCUPANCY, B40C_MIN(CompactDownsweep::CTA_OCCUPANCY, ExpandAtomicSweep::CTA_OCCUPANCY));
		int grid_size = MaxGridSize(min_occupancy, max_grid_size);

		// Make sure our spine is big enough
		int spine_elements = grid_size;
		if (retval = spine.Setup<SizeT>(grid_size, spine_elements)) exit(1);


		printf("DEBUG: BFS min occupancy %d, level-grid size %d\n", min_occupancy, grid_size);

		VertexId iteration = 0;
		SizeT queue_length;

		while (true) {

			// BFS iteration
			expand_atomic::SweepKernel<ExpandAtomicSweep><<<grid_size, ExpandAtomicSweep::THREADS>>>(
				src,
				iteration,
				bfs_problem.d_queue[0],
				bfs_problem.d_queue[1],
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_path,
				this->work_progress);

			iteration++;

			this->work_progress.GetQueueLength(iteration, queue_length);
			printf("Iteration %d BFS queued %lld elements\n",
				iteration - 1, (long long) queue_length);
			if (!queue_length) {
				break;
			}

			// Upsweep compact
			compact::UpsweepKernel<CompactUpsweep><<<grid_size, CompactUpsweep::THREADS>>>(
				iteration,
				bfs_problem.d_queue[1],
				bfs_problem.d_keep,
				(SizeT *) this->spine(),
				bfs_problem.d_collision_cache,
				this->work_progress);

			// Spine
			scan::SpineKernel<CompactSpine><<<1, CompactSpine::THREADS>>>(
				(SizeT*) spine(), (SizeT*) spine(), spine_elements);

			// Downsweep
			compact::DownsweepKernel<CompactDownsweep><<<grid_size, CompactDownsweep::THREADS>>>(
				iteration,
				bfs_problem.d_queue[1],
				bfs_problem.d_keep,
				bfs_problem.d_queue[0],
				(SizeT *) this->spine(),
				this->work_progress);
/*
			this->work_progress.GetQueueLength(iteration, queue_length);

			printf("Iteration %d compact queued %lld elements\n",
				iteration - 1, (long long) queue_length);

			if (!queue_length) {
				break;
			}
*/
		}





/*
		while (true) {

			// Contract-expand strategy
			BfsLevelGridKernel<VertexId, CollisionMask, CONTRACT_EXPAND><<<this->max_grid_size, CTA_THREADS>>>(
				src,
				bfs_problem.d_collision_cache,
				this->d_queue[queue_idx],
				this->d_queue[queue_idx ^ 1],
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_path,
				this->d_queue_lengths,
				iteration);

			if (DEBUG && cudaThreadSynchronize()) {
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

			// Upsweep
			bfs::compact::UpsweepKernel<Upsweep><<<this->max_grid_size, Upsweep::THREADS>>>(
				this->d_queue[queue_idx],
				this->d_keep,
				(SizeT *) this->spine(),
				bfs_problem.d_collision_cache);

			// Spine
			scan::SpineKernel<Spine><<<1, Spine::THREADS>>>(
				(SizeT*) spine(), (SizeT*) spine(), spine_elements);

			// Downsweep
			bfs::compact::DownsweepKernel<Downsweep><<<this->max_grid_size, Downsweep::THREADS>>>(
				this->d_queue[queue_idx],
				this->d_keep,
				this->d_queue_lengths + outgoing_queue_length_idx,
				this->d_queue[queue_idx ^ 1],
				(SizeT *) this->spine());

			queue_idx ^= 1;

			iteration++;
		}
*/

		return retval;
	}
    
};




} // namespace bfs
} // namespace b40c

