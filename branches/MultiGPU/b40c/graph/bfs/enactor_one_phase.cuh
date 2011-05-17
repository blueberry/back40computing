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
 * Single-grid compact-expand BFS implementation
 ******************************************************************************/

#pragma once

#include <bfs_base.cu>

#include <b40c/util/global_barrier.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/bfs/problem_type.cuh>

#include <b40c/bfs/compact_expand_atomic/kernel_config.cuh>
#include <b40c/bfs/compact_expand_atomic/kernel.cuh>


namespace b40c {
namespace graph {
namespace bfs {



/**
 * Single-grid breadth-first-search enactor.
 */
class SingleGridBfsEnactor : public BaseBfsEnactor
{

protected:

	/**
	 * Mechanism for implementing software global barriers from within
	 * a single grid invocation
	 */
	util::GlobalBarrierLifetime 		global_barrier;

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime 	kernel_stats;
	long long 							total_avg_live;
	long long 							total_max_live;
	long long 							total_queued;

	volatile long long 					*iteration;
	long long 							*d_iteration;

public: 	
	
	/**
	 * Constructor
	 */
	SingleGridBfsEnactor(bool DEBUG = false) :
		BaseBfsEnactor(DEBUG),
		iteration(NULL),
		d_iteration(NULL)
	{
		int flags = cudaHostAllocMapped;

		// Allocate pinned memory
		if (util::B40CPerror(cudaHostAlloc((void **)&iteration, sizeof(long long) * 1, flags),
			"SingleGridBfsEnactor cudaHostAlloc iteration failed", __FILE__, __LINE__)) exit(1);

		// Map into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_iteration, (void *) iteration, 0),
			"SingleGridBfsEnactor cudaHostGetDevicePointer iteration failed", __FILE__, __LINE__)) exit(1);
	}


	/**
	 * Destructor
	 */
	virtual ~SingleGridBfsEnactor()
	{
		if (iteration) util::B40CPerror(cudaFreeHost((void *) iteration), "SingleGridBfsEnactor cudaFreeHost iteration failed", __FILE__, __LINE__);
	}


    /**
     * Obtain statistics about the last BFS search enacted 
     */
	template <typename VertexId>
    void GetStatistics(
		long long &total_queued,
		VertexId &search_depth,
		double &avg_live)
    {
    	total_queued = this->total_queued;
    	search_depth = iteration[0] - 1;
    	avg_live = double(total_avg_live) / total_max_live;
    }
    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <bool INSTRUMENT, typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&bfs_problem,
		typename CsrProblem::VertexId 	src,
		int 								max_grid_size = 0)
	{
		cudaError_t retval = cudaSuccess;

		typedef typename CsrProblem::SizeT SizeT;
		typedef typename CsrProblem::VertexId VertexId;

		// Single-grid tuning configuration
		typedef compact_expand_atomic::KernelConfig<
			typename CsrProblem::ProblemType,
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
			6> KernelConfig;

		int occupancy = KernelConfig::CTA_OCCUPANCY;
		int grid_size = MaxGridSize(occupancy, max_grid_size);

		if (DEBUG) printf("DEBUG: BFS occupancy %d, grid size %d\n", occupancy, grid_size);
		fflush(stdout);

		// Make barriers are initialized
		if (retval = global_barrier.Setup(grid_size)) (exit(1));

		// Make sure our runtime stats are good
		if (retval = kernel_stats.Setup(grid_size)) exit(1);

		// Reset statistics
		iteration[0] 		= 0;
		total_avg_live 		= 0;
		total_max_live 		= 0;
		total_queued	 	= 0;

		// Initiate single-grid kernel
		compact_expand_atomic::Kernel<KernelConfig, INSTRUMENT, 0>
				<<<grid_size, KernelConfig::THREADS>>>(
			0,
			0,
			src,

			bfs_problem.d_expand_queue,
			bfs_problem.d_expand_parent_queue,
			bfs_problem.d_compact_queue,
			bfs_problem.d_compact_parent_queue,

			bfs_problem.d_column_indices,
			bfs_problem.d_row_offsets,
			bfs_problem.d_source_path,
			bfs_problem.d_collision_cache,
			this->work_progress,
			this->global_barrier,

			this->kernel_stats,
			(VertexId *) d_iteration);

		if (INSTRUMENT) {
			// Get stats
			kernel_stats.Accumulate(grid_size, total_avg_live, total_max_live, total_queued);
		}

		return retval;
	}
    
};



} // namespace bfs
} // namespace graph
} // namespace b40c
