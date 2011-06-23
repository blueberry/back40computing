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
 * Two-phase out-of-core BFS implementation (BFS level grid launch)
 ******************************************************************************/

#pragma once

#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/problem_type.cuh>
#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/compact_atomic/kernel.cuh>
#include <b40c/graph/bfs/compact_atomic/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Two-phase out-of-core BFS implementation (BFS level grid launch)
 *  
 * Each iteration is performed by its own kernel-launch.  For each BFS
 * iteration, two separate kernels are launched to respectively perform
 * (1) the visited-vertex culling (compaction) phase and (2) the expands
 * neighbor list expansion phase, separately.
 */
class EnactorTwoPhase : public EnactorBase
{

protected:

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime expand_kernel_stats;
	util::KernelRuntimeStatsLifetime compact_kernel_stats;
	unsigned long long 		total_avg_live;			// Running aggregate of average clock cycles per CTA (reset each traversal)
	unsigned long long 		total_max_live;			// Running aggregate of maximum clock cycles (reset each traversal)
	unsigned long long 		total_queued;
	unsigned long long 		search_depth;

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
	EnactorTwoPhase(bool DEBUG = false) :
		EnactorBase(DEBUG),
		search_depth(0),
		total_queued(0),
		done(NULL),
		d_done(NULL)
	{}


	/**
	 * Search setup / lazy initialization
	 */
	cudaError_t Setup(int expand_grid_size, int compact_grid_size)
    {
    	cudaError_t retval = cudaSuccess;

		do {

			if (!done) {
				int flags = cudaHostAllocMapped;

				// Allocate pinned memory for done
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
					"EnactorTwoPhase cudaHostAlloc done failed", __FILE__, __LINE__)) break;

				// Map done into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
					"EnactorTwoPhase cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

				// Create throttle event
				if (retval = util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
					"EnactorTwoPhase cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;
			}

			// Make sure our runtime stats are good
			if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
			if (retval = compact_kernel_stats.Setup(compact_grid_size)) break;

			// Reset statistics
			done[0] 			= 0;
			total_avg_live 		= 0;
			total_max_live 		= 0;
			total_queued 		= 0;
			search_depth 		= 0;

		} while (0);

		return retval;
	}


	/**
	 * Destructor
	 */
	virtual ~EnactorTwoPhase()
	{
		if (done) {
			util::B40CPerror(cudaFreeHost((void *) done),
					"EnactorTwoPhase cudaFreeHost done failed", __FILE__, __LINE__);

			util::B40CPerror(cudaEventDestroy(throttle_event),
				"EnactorTwoPhase cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
		}
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
    template <
    	typename ExpandPolicy,
    	typename CompactPolicy,
    	bool INSTRUMENT,
    	typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::VertexId					VertexId;
		typedef typename CsrProblem::SizeT						SizeT;

		cudaError_t retval = cudaSuccess;

		do {
			// Determine grid size(s)
			int expand_min_occupancy 		= ExpandPolicy::CTA_OCCUPANCY;
			int expand_grid_size 			= MaxGridSize(expand_min_occupancy, max_grid_size);

			int compact_min_occupancy		= CompactPolicy::CTA_OCCUPANCY;
			int compact_grid_size 			= MaxGridSize(compact_min_occupancy, max_grid_size);

			if (DEBUG) printf("BFS expand min occupancy %d, level-grid size %d\n",
				expand_min_occupancy, expand_grid_size);
			if (DEBUG) printf("BFS compact min occupancy %d, level-grid size %d\n",
				compact_min_occupancy, compact_grid_size);

			if (INSTRUMENT) {
				printf("Compaction queue, Expansion queue\n");
				printf("1, ");
			}

			SizeT queue_length;
			VertexId iteration = 0;		// BFS iteration
			VertexId queue_index = 0;	// Work stealing/queue index

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Setup / lazy initialization
			if (retval = Setup(expand_grid_size, compact_grid_size)) break;

			while (!done[0]) {

				int selector = queue_index & 1;

				// Expansion
				expand_atomic::Kernel<ExpandPolicy>
					<<<expand_grid_size, ExpandPolicy::THREADS>>>(
						src,
						1,											// num_elements (unused if not first iteration: we obtain this from device-side counters instead)
						iteration,
						queue_index,
						queue_index,								// also serves as steal_index
						1,											// number of GPUs
						d_done,
						graph_slice->frontier_queues.d_keys[selector],
						graph_slice->frontier_queues.d_keys[selector ^ 1],
						graph_slice->frontier_queues.d_values[selector],
						graph_slice->frontier_queues.d_values[selector ^ 1],
						graph_slice->d_column_indices,
						graph_slice->d_row_offsets,
						graph_slice->d_source_path,
						this->work_progress,
						this->expand_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;

				queue_index++;
				iteration++;

				if (INSTRUMENT) {
					// Get expansion queue length
					if (work_progress.GetQueueLength(queue_index, queue_length)) break;
					total_queued += queue_length;
					printf("%lld\n", (long long) queue_length);

					// Get expand stats (i.e., duty %)
					if (retval = expand_kernel_stats.Accumulate(expand_grid_size, total_avg_live, total_max_live)) break;
				}

				// Compaction
				compact_atomic::Kernel<CompactPolicy>
					<<<compact_grid_size, CompactPolicy::THREADS>>>(
						0,											// num_elements (unused: we obtain this from device-side counters instead)
						queue_index,
						queue_index,								// also serves as steal_index
						d_done,
						graph_slice->frontier_queues.d_keys[selector ^ 1],
						graph_slice->frontier_queues.d_keys[selector],
						graph_slice->frontier_queues.d_values[selector ^ 1],
						graph_slice->frontier_queues.d_values[selector],
						graph_slice->d_collision_cache,
						this->work_progress,
						this->compact_kernel_stats);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "compact_atomic::Kernel failed ", __FILE__, __LINE__))) break;

				queue_index++;

				if (INSTRUMENT) {
					// Get compaction queue length
					if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
					printf("%lld, ", (long long) queue_length);

					// Get compact downsweep stats (i.e., duty %)
					if (retval = compact_kernel_stats.Accumulate(compact_grid_size, total_avg_live, total_max_live)) break;
				}

				// Throttle
				if (iteration & 1) {
					if (retval = util::B40CPerror(cudaEventRecord(throttle_event),
						"EnactorTwoPhase cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
				} else {
					if (retval = util::B40CPerror(cudaEventSynchronize(throttle_event),
						"EnactorTwoPhase cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
				};
			}
			if (retval) break;

		} while(0);

		return retval;
	}


    /**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <bool INSTRUMENT, typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		if (this->cuda_props.device_sm_version >= 200) {

			// Expansion kernel config
			typedef expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				INSTRUMENT, 			// INSTRUMENT
				0, 						// SATURATION_QUIT
				true, 					// DEQUEUE_PROBLEM_SIZE
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				0,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::cg,		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::cg,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::cg,		// QUEUE_WRITE_MODIFIER,
				true,					// WORK_STEALING
				6> ExpandPolicy;

			// Compaction kernel config
			typedef compact_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				INSTRUMENT, 			// INSTRUMENT
				true, 					// DEQUEUE_PROBLEM_SIZE
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				2,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				9> CompactPolicy;

			return EnactSearch<ExpandPolicy, CompactPolicy, INSTRUMENT>(
				csr_problem, src, max_grid_size);

		} else {
			printf("Not yet tuned for this architecture\n");
			return cudaErrorInvalidConfiguration;
		}
	}
    
};



} // namespace bfs
} // namespace graph
} // namespace b40c
