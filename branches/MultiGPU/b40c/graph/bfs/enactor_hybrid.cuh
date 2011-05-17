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

#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/bfs/problem_type.cuh>
#include <b40c/bfs/compact_expand_atomic/kernel.cuh>
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
class HybridBfsEnactor : public BaseBfsEnactor
{

protected:

	/**
	 * Mechanism for implementing software global barriers from within
	 * a single grid invocation
	 */
	util::GlobalBarrierLifetime global_barrier;

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime 	single_kernel_stats;
	util::KernelRuntimeStatsLifetime 	expand_kernel_stats;
	util::KernelRuntimeStatsLifetime 	compact_kernel_stats;
	long long 							total_avg_live;			// Running aggregate of average clock cycles per CTA (reset each traversal)
	long long 							total_max_live;			// Running aggregate of maximum clock cycles (reset each traversal)
	long long 							total_queued;
	long long 							search_depth;

	/**
	 * Throttle state.  We want the host to have an additional BFS iteration
	 * of kernel launches queued up for for pipeline efficiency (particularly on
	 * Windows), so we keep a pinned, mapped word that the traversal kernels will
	 * signal when done.
	 */
	volatile int 		*done;
	int 				*d_done;
	cudaEvent_t			throttle_event;

	/**
	 * Iteration
	 */
	volatile long long 	*iteration;
	long long 			*d_iteration;

public: 	
	
	/**
	 * Constructor
	 */
	HybridBfsEnactor(bool DEBUG = false) :
		BaseBfsEnactor(DEBUG),
		search_depth(0),
		total_queued(0),
		done(NULL),
		d_done(NULL)
	{
		int flags = cudaHostAllocMapped;

		// Allocate pinned memory for done
		if (util::B40CPerror(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
			"HybridBfsEnactor cudaHostAlloc done failed", __FILE__, __LINE__)) exit(1);

		// Map done into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
			"HybridBfsEnactor cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) exit(1);

		// Create throttle event
		if (util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
			"HybridBfsEnactor cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) exit(1);

		// Allocate pinned memory for iteration
		if (util::B40CPerror(cudaHostAlloc((void **)&iteration, sizeof(long long) * 1, flags),
			"HybridBfsEnactor cudaHostAlloc iteration failed", __FILE__, __LINE__)) exit(1);

		// Map iteration into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_iteration, (void *) iteration, 0),
			"HybridBfsEnactor cudaHostGetDevicePointer d_iteration failed", __FILE__, __LINE__)) exit(1);
	}


	/**
	 * Destructor
	 */
	virtual ~HybridBfsEnactor()
	{
		if (done) util::B40CPerror(cudaFreeHost((void *) done), "HybridBfsEnactor cudaFreeHost done failed", __FILE__, __LINE__);
		util::B40CPerror(cudaEventDestroy(throttle_event), "HybridBfsEnactor cudaEventDestroy throttle_event failed", __FILE__, __LINE__);

		if (iteration) util::B40CPerror(cudaFreeHost((void *) iteration), "HybridBfsEnactor cudaFreeHost iteration failed", __FILE__, __LINE__);
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
			6> SingleConfig;

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

		int single_min_occupancy 		= SingleConfig::CTA_OCCUPANCY;
		int single_grid_size 			= MaxGridSize(single_min_occupancy, max_grid_size);

		int expand_min_occupancy 		= ExpandConfig::CTA_OCCUPANCY;
		int expand_grid_size 			= MaxGridSize(expand_min_occupancy, max_grid_size);

		int compact_min_occupancy		= CompactConfig::CTA_OCCUPANCY;
		int compact_grid_size 			= MaxGridSize(compact_min_occupancy, max_grid_size);

		if (DEBUG) printf("BFS single min occupancy %d, level-grid size %d\n",
				single_min_occupancy, single_grid_size);
		if (DEBUG) printf("BFS expand min occupancy %d, level-grid size %d\n",
				expand_min_occupancy, expand_grid_size);
		if (DEBUG) printf("BFS compact min occupancy %d, level-grid size %d\n",
			compact_min_occupancy, compact_grid_size);

		// Make sure barriers are initialized
		if (retval = global_barrier.Setup(single_grid_size)) (exit(1));

		// Make sure our runtime stats are good
		if (retval = single_kernel_stats.Setup(single_grid_size)) exit(1);
		if (retval = expand_kernel_stats.Setup(expand_grid_size)) exit(1);
		if (retval = compact_kernel_stats.Setup(compact_grid_size)) exit(1);

		// Reset statistics
		total_queued 		= 0;
		total_avg_live 		= 0;
		total_max_live 		= 0;
		iteration[0] 		= 0;

		SizeT queue_length 				= 0;
		int selector 					= 0;
		VertexId *d_queues[2] 			= {bfs_problem.d_compact_queue, 		bfs_problem.d_expand_queue};
		VertexId *d_parent_queues[2] 	= {bfs_problem.d_compact_parent_queue, 	bfs_problem.d_expand_parent_queue};

		if (INSTRUMENT) {
			printf("1, 1\n");
		}

		const int SATURATION_QUIT = 4;

		VertexId queue_index = 0;

		do {

			VertexId phase_iteration = iteration[0];

			if (queue_length <= single_grid_size * SingleConfig::TILE_ELEMENTS * SATURATION_QUIT) {

				// Run single-grid, no-separate-compaction
				compact_expand_atomic::Kernel<SingleConfig, INSTRUMENT, SATURATION_QUIT>
						<<<single_grid_size, SingleConfig::THREADS>>>(
					iteration[0],
					queue_index,
					src,

					d_queues[selector],
					d_parent_queues[selector],
					d_queues[selector ^ 1],
					d_parent_queues[selector ^ 1],

					bfs_problem.d_column_indices,
					bfs_problem.d_row_offsets,
					bfs_problem.d_source_path,
					bfs_problem.d_collision_cache,
					this->work_progress,
					this->global_barrier,

					this->single_kernel_stats,
					(VertexId *) d_iteration);

				// Synchronize to make sure we have a coherent iteration;
				cudaThreadSynchronize();

				if ((iteration[0] - phase_iteration) & 1) {
					// An odd number of iterations passed: update selector
					selector ^= 1;
				}
				// Update queue index by the number of elapsed iterations
				queue_index += (iteration[0] - phase_iteration);

				// Get queue length
				if (this->work_progress.GetQueueLength(iteration[0], queue_length)) exit(0);

				if (INSTRUMENT) {
					// Get stats
					single_kernel_stats.Accumulate(single_grid_size, total_avg_live, total_max_live, total_queued);
					total_queued += queue_length;
					printf("%lld, %lld\n", iteration[0], (long long) queue_length);
				}

			} else {

				done[0] = 0;
				while (!done[0]) {

					// Run level-grid

					// Compaction
					compact_atomic::Kernel<CompactConfig, INSTRUMENT><<<compact_grid_size, CompactConfig::THREADS>>>(
						queue_index,
						d_done,
						d_queues[selector],						// in
						d_parent_queues[selector],
						d_queues[selector ^ 1],					// out
						d_parent_queues[selector ^ 1],
						bfs_problem.d_collision_cache,
						this->work_progress,
						this->compact_kernel_stats);

					queue_index++;

					if (INSTRUMENT) {
						// Get compact downsweep stats (i.e., duty %)
						if (this->work_progress.GetQueueLength(iteration[0], queue_length)) exit(0);
						printf("%lld, %lld", iteration[0], (long long) queue_length);
						if (compact_kernel_stats.Accumulate(compact_grid_size, total_avg_live, total_max_live)) exit(0);
					}

					// Expansion
					expand_atomic::Kernel<ExpandConfig, INSTRUMENT, 0>
							<<<expand_grid_size, ExpandConfig::THREADS>>>(
						src,
						(VertexId) iteration[0],
						queue_index,
						d_done,
						d_queues[selector ^ 1],
						d_parent_queues[selector ^ 1],
						d_queues[selector],
						d_parent_queues[selector],
						bfs_problem.d_column_indices,
						bfs_problem.d_row_offsets,
						bfs_problem.d_source_path,
						this->work_progress,
						this->expand_kernel_stats);

					queue_index++;
					iteration[0]++;

					if (INSTRUMENT) {
						// Get expand stats (i.e., duty %)
						expand_kernel_stats.Accumulate(expand_grid_size, total_avg_live, total_max_live);
						if (this->work_progress.GetQueueLength(iteration[0], queue_length)) exit(0);
						total_queued += queue_length;
						printf(", %lld\n", (long long) queue_length);
					}

					// Throttle
					if ((iteration[0] - phase_iteration) & 1) {
						if (util::B40CPerror(cudaEventRecord(throttle_event),
							"LevelGridBfsEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) exit(1);
					} else {
						if (util::B40CPerror(cudaEventSynchronize(throttle_event),
							"LevelGridBfsEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) exit(1);
					};
				}

				// Get queue length
				if (this->work_progress.GetQueueLength(iteration[0], queue_length)) exit(0);
			}

		} while (queue_length > 0);

		search_depth = iteration[0] - 1;
		printf("\n");

		return retval;
	}
    
};


} // namespace bfs
} // namespace graph
} // namespace b40c
