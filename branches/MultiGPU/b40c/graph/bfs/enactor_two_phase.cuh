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
 * Level-grid compact-expand BFS implementation
 ******************************************************************************/

#pragma once

#include <vector>

#include <bfs_base.cu>

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/bfs/problem_type.cuh>
#include <b40c/bfs/expand_atomic/kernel.cuh>
#include <b40c/bfs/expand_atomic/kernel_config.cuh>
#include <b40c/bfs/compact_atomic/kernel.cuh>
#include <b40c/bfs/compact_atomic/kernel_config.cuh>

namespace b40c {
namespace graph {
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
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime expand_kernel_stats;
	util::KernelRuntimeStatsLifetime compact_kernel_stats;
	long long 		total_avg_live;			// Running aggregate of average clock cycles per CTA (reset each traversal)
	long long 		total_max_live;			// Running aggregate of maximum clock cycles (reset each traversal)
	long long 		total_queued;
	long long 		search_depth;

	/**
	 * Throttle state.  We want the host to have an additional BFS iteration
	 * of kernel launches queued up for for pipeline efficiency (particularly on
	 * Windows), so we keep a pinned, mapped word that the traversal kernels will
	 * signal when done.
	 */
	volatile int 	*done;
	int 			*d_done;
	cudaEvent_t		throttle_event;

public: 	
	
	/**
	 * Constructor
	 */
	LevelGridBfsEnactor(bool DEBUG = false) :
		BaseBfsEnactor(DEBUG),
		search_depth(0),
		total_queued(0),
		done(NULL),
		d_done(NULL)
	{
		int flags = cudaHostAllocMapped;

		// Allocate pinned memory for done
		if (util::B40CPerror(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
			"LevelGridBfsEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) exit(1);

		// Map done into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
			"LevelGridBfsEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) exit(1);

		// Create throttle event
		if (util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
			"LevelGridBfsEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) exit(1);
	}


	/**
	 * Destructor
	 */
	virtual ~LevelGridBfsEnactor()
	{
		if (done) util::B40CPerror(cudaFreeHost((void *) done), "LevelGridBfsEnactor cudaFreeHost done failed", __FILE__, __LINE__);
		util::B40CPerror(cudaEventDestroy(throttle_event), "LevelGridBfsEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
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
    	search_depth = this->search_depth;
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
		// Expansion kernel config
		typedef expand_atomic::KernelConfig<
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
			true,					// WORK_STEALING
			6> ExpandConfig;

		// Compaction kernel config
		typedef compact_atomic::KernelConfig<
			typename CsrProblem::ProblemType,
			200,
			8,
			7,
			0,
			2,
			5,
			util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
			util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
			false,					// WORK_STEALING
			9> CompactConfig;

		typedef typename CsrProblem::VertexId					VertexId;
		typedef typename CsrProblem::SizeT						SizeT;

		cudaError_t retval = cudaSuccess;

		//
		// Determine grid size(s)
		//

		int expand_min_occupancy 		= ExpandConfig::CTA_OCCUPANCY;
		int expand_grid_size 			= MaxGridSize(expand_min_occupancy, max_grid_size);
		int compact_min_occupancy		= CompactConfig::CTA_OCCUPANCY;
		int compact_grid_size 			= MaxGridSize(compact_min_occupancy, max_grid_size);

		if (DEBUG) printf("BFS expand min occupancy %d, level-grid size %d\n",
			expand_min_occupancy, expand_grid_size);
		if (DEBUG) printf("BFS compact min occupancy %d, level-grid size %d\n",
			compact_min_occupancy, compact_grid_size);

		// Make sure our runtime stats are good
		if (retval = expand_kernel_stats.Setup(expand_grid_size)) exit(1);
		if (retval = compact_kernel_stats.Setup(compact_grid_size)) exit(1);

		// Reset statistics
		total_queued 		= 0;
		done[0] 			= 0;
		total_avg_live 		= 0;
		total_max_live 		= 0;

		if (INSTRUMENT) {
			printf("1, 1\n");
		}

		SizeT queue_length;
		VertexId iteration = 0;
		VertexId queue_index = 0;

		while (!done[0]) {

			// Expansion
			expand_atomic::Kernel<ExpandConfig, INSTRUMENT, 0>
					<<<expand_grid_size, ExpandConfig::THREADS>>>(
				src,
				iteration,
				queue_index,
				d_done,
				bfs_problem.d_compact_queue,			// in
				bfs_problem.d_compact_parent_queue,
				bfs_problem.d_expand_queue,				// out
				bfs_problem.d_expand_parent_queue,
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_path,
				this->work_progress,
				this->expand_kernel_stats);

			queue_index++;
			iteration++;

			if (INSTRUMENT) {
				// Get expansion queue length
				if (this->work_progress.GetQueueLength(queue_index, queue_length)) exit(0);
				total_queued += queue_length;
				printf("Expansion queue length: %lld\n", (long long) queue_length);

				// Get expand stats (i.e., duty %)
				expand_kernel_stats.Accumulate(expand_grid_size, total_avg_live, total_max_live);
			}

			// Compaction
			compact_atomic::Kernel<CompactConfig, INSTRUMENT><<<compact_grid_size, CompactConfig::THREADS>>>(
				queue_index,
				d_done,
				bfs_problem.d_expand_queue,				// in
				bfs_problem.d_expand_parent_queue,
				bfs_problem.d_compact_queue,			// out
				bfs_problem.d_compact_parent_queue,
				bfs_problem.d_collision_cache,
				this->work_progress,
				this->compact_kernel_stats);

			queue_index++;

			if (INSTRUMENT) {
				// Get compaction queue length
				if (this->work_progress.GetQueueLength(iteration, queue_length)) exit(0);
				printf("Compaction queue length: %lld\n", (long long) queue_length);

				// Get compact downsweep stats (i.e., duty %)
				if (compact_kernel_stats.Accumulate(compact_grid_size, total_avg_live, total_max_live)) exit(0);
			}

			// Throttle
			if (iteration & 1) {
				if (util::B40CPerror(cudaEventRecord(throttle_event),
					"LevelGridBfsEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) exit(1);
			} else {
				if (util::B40CPerror(cudaEventSynchronize(throttle_event),
					"LevelGridBfsEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) exit(1);
			};
		}
		printf("\n");

		printf("Launched iterations: %d\n", iteration);
		search_depth = iteration - 1;

		return retval;
	}
    
};



} // namespace bfs
} // namespace graph
} // namespace b40c
