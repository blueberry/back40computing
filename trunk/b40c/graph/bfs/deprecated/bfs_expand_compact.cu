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
 * Expand-compact BFS implementation
 ******************************************************************************/

#pragma once

#include <bfs_base.cu>

#include <b40c/util/spine.cuh>
#include <b40c/bfs/problem_type.cuh>

#include <b40c/bfs/expand_compact/sweep_kernel.cuh>
#include <b40c/bfs/expand_compact/sweep_kernel_config.cuh>

namespace b40c {
namespace bfs {


/**
 * Expand-compact breadth-first-search enactor.
 *  
 * Each iterations is performed by its own kernel-launch.  
 */
class ExpandCompactBfsEnactor : public BaseBfsEnactor
{

protected:

	long long iteration;
	long long total_queued;

public: 	
	
	/**
	 * Constructor
	 */
	ExpandCompactBfsEnactor(bool DEBUG = false) :
		BaseBfsEnactor(DEBUG),
		iteration(0),
		total_queued(0)
			{}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_barrier_wait)		// total time spent waiting in barriers in ms (threadblock average)
    {
    	total_queued = this->total_queued;
    	search_depth = iteration - 1;
    	avg_barrier_wait = 0;
    }
    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <typename BfsCsrProblem>
	cudaError_t EnactSearch(
		BfsCsrProblem 						&bfs_problem,
		typename BfsCsrProblem::VertexId 	src,
		int 								max_grid_size = 0)
	{
		// kernel config
		typedef expand_compact::SweepKernelConfig<
			typename BfsCsrProblem::ProblemType,
			200,
			8,
			7,
			0,
			0,
			5,
			util::io::ld::cg,		// QUEUE_READ_MODIFIER,
			util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
			util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
			util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
			util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
			false,					// WORK_STEALING
			6> BfsAtomicSweep;


		typedef typename BfsCsrProblem::VertexId					VertexId;
		typedef typename BfsCsrProblem::SizeT						SizeT;

		//
		// Determine grid size(s)
		//

		int expand_min_occupancy = BfsAtomicSweep::CTA_OCCUPANCY;
		int expand_grid_size = MaxGridSize(expand_min_occupancy, max_grid_size);

		printf("DEBUG: BFS expand min occupancy %d, level-grid size %d\n",
				expand_min_occupancy, expand_grid_size);

		cudaError_t retval = cudaSuccess;

		iteration = 0;
		total_queued = 0;
		SizeT queue_length;

		printf("Iteration, Expand\n");
		while (true) {

			// BFS iteration
			expand_compact::SweepKernel<BfsAtomicSweep><<<expand_grid_size, BfsAtomicSweep::THREADS>>>(
				src,
				iteration,
				bfs_problem.d_compact_queue,			// in
				bfs_problem.d_expand_queue,				// out
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_path,
				bfs_problem.d_collision_cache,
				this->work_progress);

			iteration++;

			this->work_progress.GetQueueLength(iteration, queue_length);
			total_queued += queue_length;
			printf("%lld, %lld\n", iteration, (long long) queue_length);
			if (!queue_length) {
				break;
			}

			// BFS iteration
			expand_compact::SweepKernel<BfsAtomicSweep><<<expand_grid_size, BfsAtomicSweep::THREADS>>>(
				src,
				iteration,
				bfs_problem.d_expand_queue,				// out
				bfs_problem.d_compact_queue,			// in
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_path,
				bfs_problem.d_collision_cache,
				this->work_progress);

			iteration++;

			this->work_progress.GetQueueLength(iteration, queue_length);
			total_queued += queue_length;
			printf("%lld, %lld\n", iteration, (long long) queue_length);
			if (!queue_length) {
				break;
			}
		}

		printf("\n");

		return retval;
	}
    
};




} // namespace bfs
} // namespace b40c

