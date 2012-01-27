/******************************************************************************
 * Copyright 2010-2012 Duane Merrill
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
 * Hybrid BFS enactor
 ******************************************************************************/

#pragma once

#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/problem_type.cuh>

#include <b40c/graph/bfs/contract_expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/contract_expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/expand_atomic/kernel_policy.cuh>
#include <b40c/graph/bfs/two_phase/contract_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/contract_atomic/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Hybrid BFS enactor.
 *  
 * Combines functionality of contract-expand and two-phase enactors,
 * running contract-expand (only global edge frontier) for small-sized
 * BFS iterations and two-phase (global edge and vertex frontiers) for
 * large-sized BFS iterations.
 */
template <bool INSTRUMENT>							// Whether or not to collect per-CTA clock-count statistics
class EnactorHybrid : public EnactorBase
{

	//---------------------------------------------------------------------
	// Members
	//---------------------------------------------------------------------

protected:

	/**
	 * Mechanism for implementing software global barriers from within
	 * a fused grid invocation
	 */
	util::GlobalBarrierLifetime global_barrier;

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime 	fused_kernel_stats;
	util::KernelRuntimeStatsLifetime 	expand_kernel_stats;
	util::KernelRuntimeStatsLifetime 	contract_kernel_stats;
	unsigned long long 					total_runtimes;			// Total time "worked" by each cta
	unsigned long long 					total_lifetimes;		// Total time elapsed by each cta
	unsigned long long 					total_queued;

	/**
	 * Throttle state.  We want the host to have an additional BFS h_iteration
	 * of kernel launches queued up for for pipeline efficiency (particularly on
	 * Windows), so we keep a pinned, mapped word that the traversal kernels will
	 * signal when done.
	 */
	volatile int 		*done;
	int 				*d_done;
	cudaEvent_t			throttle_event;

	/**
	 * Iteration output (from fused kernels)
	 */
	long long 			*d_iteration;
	long long 			h_iteration;

	
	//---------------------------------------------------------------------
	// Methods
	//---------------------------------------------------------------------

protected:

	/**
	 * Prepare enactor for search.  Must be called prior to each search.
	 */
	template <typename CsrProblem>
	cudaError_t Setup(
		CsrProblem &csr_problem,
		int fused_grid_size,
		int expand_grid_size,
		int contract_grid_size)
    {
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;
		do {

			if (!done) {
				int flags = cudaHostAllocMapped;

				// Allocate pinned memory for done
				if (retval = util::B40CPerror(cudaHostAlloc((void **)&done, sizeof(int) * 1, flags),
					"EnactorHybrid cudaHostAlloc done failed", __FILE__, __LINE__)) break;

				// Map done into GPU space
				if (retval = util::B40CPerror(cudaHostGetDevicePointer((void **)&d_done, (void *) done, 0),
					"EnactorHybrid cudaHostGetDevicePointer done failed", __FILE__, __LINE__)) break;

				// Create throttle event
				if (retval = util::B40CPerror(cudaEventCreateWithFlags(&throttle_event, cudaEventDisableTiming),
					"EnactorHybrid cudaEventCreateWithFlags throttle_event failed", __FILE__, __LINE__)) break;

				// Allocate gpu memory for d_iteration
				if (retval = util::B40CPerror(cudaMalloc((void**) &d_iteration, sizeof(long long)),
					"EnactorHybrid cudaMalloc d_iteration failed", __FILE__, __LINE__)) break;
			}

			// Make sure our runtime stats are good
			if (retval = fused_kernel_stats.Setup(fused_grid_size)) break;
			if (retval = expand_kernel_stats.Setup(expand_grid_size)) break;
			if (retval = contract_kernel_stats.Setup(contract_grid_size)) break;

			// Make sure barriers are initialized
			if (retval = global_barrier.Setup(fused_grid_size)) break;

			// Reset statistics
			done[0] 			= -1;
			total_runtimes 		= 0;
			total_lifetimes 	= 0;
			total_queued 		= 0;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			// Bind bitmask texture
			int bytes = (graph_slice->nodes + 8 - 1) / 8;
			cudaChannelFormatDesc bitmask_desc = cudaCreateChannelDesc<char>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					two_phase::contract_atomic::BitmaskTex<VisitedMask>::ref,
					graph_slice->d_visited_mask,
					bitmask_desc,
					bytes),
				"EnactorHybrid cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			cudaChannelFormatDesc row_offsets_desc = cudaCreateChannelDesc<SizeT>();
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					two_phase::expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorHybrid cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind bitmask texture
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					contract_expand_atomic::BitmaskTex<VisitedMask>::ref,
					graph_slice->d_visited_mask,
					bitmask_desc,
					bytes),
				"EnactorHybrid cudaBindTexture bitmask_tex_ref failed", __FILE__, __LINE__)) break;

			// Bind row-offsets texture
			if (retval = util::B40CPerror(cudaBindTexture(
					0,
					contract_expand_atomic::RowOffsetTex<SizeT>::ref,
					graph_slice->d_row_offsets,
					row_offsets_desc,
					(graph_slice->nodes + 1) * sizeof(SizeT)),
				"EnactorHybrid cudaBindTexture row_offset_tex_ref failed", __FILE__, __LINE__)) break;


		} while (0);

		return retval;
	}

public:

	/**
	 * Constructor
	 */
	EnactorHybrid(bool DEBUG = false) :
		EnactorBase(EDGE_FRONTIERS, DEBUG),
		d_iteration(NULL),
		h_iteration(0),
		total_queued(0),
		done(NULL),
		d_done(NULL)
	{}


	/**
	 * Destructor
	 */
	virtual ~EnactorHybrid()
	{
		if (done) {
			util::B40CPerror(cudaFreeHost((void *) done), "EnactorHybrid cudaFreeHost done failed", __FILE__, __LINE__);
			util::B40CPerror(cudaEventDestroy(throttle_event), "EnactorHybrid cudaEventDestroy throttle_event failed", __FILE__, __LINE__);
		}
		if (d_iteration) {
			util::B40CPerror(cudaFree((void *) d_iteration), "EnactorHybrid cudaFree d_iteration failed", __FILE__, __LINE__);
		}
	}


    /**
     * Obtain statistics about the last BFS search enacted
     */
	template <typename VertexId>
    void GetStatistics(
    	long long &total_queued,
    	VertexId &search_depth,
    	double &avg_duty)
    {
		cudaThreadSynchronize();

		total_queued = this->total_queued;
    	search_depth = h_iteration - 1;

    	avg_duty = (total_lifetimes > 0) ?
    		double(total_runtimes) / total_lifetimes :
    		0.0;
    }


	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <
    	typename OnePhasePolicy,
    	typename ExpandPolicy,
    	typename ContractPolicy,
    	typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{
		typedef typename CsrProblem::SizeT 			SizeT;
		typedef typename CsrProblem::VertexId 		VertexId;
		typedef typename CsrProblem::VisitedMask 	VisitedMask;

		cudaError_t retval = cudaSuccess;

		do {

			// Determine grid size(s)
			int fused_min_occupancy 	= OnePhasePolicy::CTA_OCCUPANCY;
			int fused_grid_size 		= MaxGridSize(fused_min_occupancy, max_grid_size);

			int expand_min_occupancy 		= ExpandPolicy::CTA_OCCUPANCY;
			int expand_grid_size 			= MaxGridSize(expand_min_occupancy, max_grid_size);

			int contract_min_occupancy		= ContractPolicy::CTA_OCCUPANCY;
			int contract_grid_size 			= MaxGridSize(contract_min_occupancy, max_grid_size);

			if (DEBUG) {
				printf("BFS fused min occupancy %d, level-grid size %d\n",
					fused_min_occupancy, fused_grid_size);
				printf("BFS expand min occupancy %d, level-grid size %d\n",
					expand_min_occupancy, expand_grid_size);
				printf("BFS contract min occupancy %d, level-grid size %d\n",
					contract_min_occupancy, contract_grid_size);

				printf("Iteration, Queue Size\n");
				printf("1, 1\n");
			}

			// Lazy initialization
			if (retval = Setup(
				csr_problem,
				fused_grid_size,
				expand_grid_size,
				contract_grid_size)) break;

			// Single-gpu graph slice
			typename CsrProblem::GraphSlice *graph_slice = csr_problem.graph_slices[0];

			VertexId iteration 				= 0;
			VertexId queue_index 			= 0;	// Work stealing/queue index
			SizeT queue_length 				= 0;
			int selector 					= 0;

			while (done[0] != 0) {

				VertexId phase_iteration = iteration;

				// Run fused contract-expand kernel
				contract_expand_atomic::KernelGlobalBarrier<OnePhasePolicy>
					<<<fused_grid_size, OnePhasePolicy::THREADS>>>(
						iteration,
						queue_index,
						queue_index,												// also serves as steal_index
						src,
						graph_slice->frontier_queues.d_keys[selector],				// in edge frontier
						graph_slice->frontier_queues.d_keys[selector ^ 1],			// out edge frontier
						graph_slice->frontier_queues.d_values[selector],			// in predecessors
						graph_slice->frontier_queues.d_values[selector ^ 1],		// out predecessors
						graph_slice->d_column_indices,
						graph_slice->d_row_offsets,
						graph_slice->d_labels,
						graph_slice->d_visited_mask,
						work_progress,
						graph_slice->frontier_elements[0],							// max frontier vertices (all queues should be the same size)
						global_barrier,
						fused_kernel_stats,
						(VertexId *) d_iteration);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_expand_atomic::KernelGlobalBarrier failed ", __FILE__, __LINE__))) break;

				// Retrieve output iteration
				if (retval = util::B40CPerror(cudaMemcpy(
					&iteration,
					(VertexId *) d_iteration,
					sizeof(VertexId),
					cudaMemcpyDeviceToHost),
						"EnactorHybrid cudaMemcpy d_iteration failed", __FILE__, __LINE__)) break;

				// Check if done or just saturated
				if (iteration < 0) {
					iteration *= -1;			// saturated
				} else {
					break;						// done
				}

				if ((iteration - phase_iteration) & 1) {
					// An odd number of iterations passed: update selector
					selector ^= 1;
				}
				// Update queue index by the number of elapsed iterations
				queue_index += (iteration - phase_iteration);

				if (DEBUG) {
					// Get queue length
					if (retval = work_progress.GetQueueLength(queue_index, queue_length)) break;
					printf("%lld, %lld\n", (long long) iteration, (long long) queue_length);
				}
				if (INSTRUMENT) {
					// Get stats
					if (retval = fused_kernel_stats.Accumulate(
						fused_grid_size,
						total_runtimes,
						total_lifetimes,
						total_queued)) break;
				}

				// Run two-phase until done is not -1
				done[0] = -1;

				while (done[0] < 0) {

					// Contraction
					two_phase::contract_atomic::Kernel<ContractPolicy>
						<<<contract_grid_size, ContractPolicy::THREADS>>>(
							src,
							iteration,
							0,														// num_elements (unused: we obtain this from device-side counters instead)
							queue_index,
							queue_index,											// also serves as steal_index
							1,														// number of GPUs
							d_done,
							graph_slice->frontier_queues.d_keys[selector],			// in edge frontier
							graph_slice->frontier_queues.d_keys[selector ^ 1],		// out vertex frontier
							graph_slice->frontier_queues.d_values[selector],		// in predecessors
							graph_slice->d_labels,
							graph_slice->d_visited_mask,
							work_progress,
							graph_slice->frontier_elements[selector],				// max in vertices
							graph_slice->frontier_elements[selector ^ 1],			// max out vertices
							contract_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "contract_atomic::Kernel failed ", __FILE__, __LINE__))) break;

					queue_index++;

					if (DEBUG) {
						// Get contract downsweep stats (i.e., duty %)
						if (work_progress.GetQueueLength(queue_index, queue_length)) break;
						printf("%lld, %lld", (long long) iteration, (long long) queue_length);
					}
					if (INSTRUMENT) {
						if (contract_kernel_stats.Accumulate(
							contract_grid_size,
							total_runtimes,
							total_lifetimes)) break;
					}

					// Expansion
					two_phase::expand_atomic::Kernel<ExpandPolicy>
						<<<expand_grid_size, ExpandPolicy::THREADS>>>(
							queue_index,
							queue_index,											// also serves as steal_index
							1,														// number of GPUs
							d_done,
							graph_slice->frontier_queues.d_keys[selector ^ 1],		// in vertex frontier
							graph_slice->frontier_queues.d_keys[selector],			// out edge frontier
							graph_slice->frontier_queues.d_values[selector],		// out predecessors
							graph_slice->d_column_indices,
							graph_slice->d_row_offsets,
							work_progress,
							graph_slice->frontier_elements[selector ^ 1],			// max in vertices
							graph_slice->frontier_elements[selector],				// max out vertices
							expand_kernel_stats);

					if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "expand_atomic::Kernel failed ", __FILE__, __LINE__))) break;

					queue_index++;
					iteration++;

					if (INSTRUMENT || DEBUG) {
						if (work_progress.GetQueueLength(queue_index, queue_length)) break;
						total_queued += queue_length;
						if (DEBUG) printf(", %lld\n", (long long) queue_length);
						if (INSTRUMENT) {
							expand_kernel_stats.Accumulate(
								expand_grid_size,
								total_runtimes,
								total_lifetimes);
						}
					}

					// Throttle
					if ((iteration - phase_iteration) & 1) {
						if (util::B40CPerror(cudaEventRecord(throttle_event),
							"LevelGridBfsEnactor cudaEventRecord throttle_event failed", __FILE__, __LINE__)) break;
					} else {
						if (util::B40CPerror(cudaEventSynchronize(throttle_event),
							"LevelGridBfsEnactor cudaEventSynchronize throttle_event failed", __FILE__, __LINE__)) break;
					}
				}
			}
			if (retval) break;

			// Check if any of the frontiers overflowed due to redundant expansion
			bool overflowed = false;
			if (retval = work_progress.CheckOverflow<SizeT>(overflowed)) break;
			if (overflowed) {
				retval = util::B40CPerror(cudaErrorInvalidConfiguration, "Frontier queue overflow.  Please increase queue-sizing factor. ", __FILE__, __LINE__);
				break;
			}

			h_iteration = iteration;

		} while (0);

		return retval;
	}
    

	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <typename CsrProblem>
	cudaError_t EnactSearch(
		CsrProblem 						&csr_problem,
		typename CsrProblem::VertexId 	src,
		int 							max_grid_size = 0)
	{

    	// GF100
    	if (cuda_props.device_sm_version >= 200) {

        	const int SATURATION_QUIT = 4 * 128;

        	// Fused-grid tuning configuration
			typedef contract_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				INSTRUMENT, 			// INSTRUMENT
				SATURATION_QUIT,		// SATURATION_QUIT
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
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				0,						// BITMASK_CULL_THRESHOLD
				6> 						// LOG_SCHEDULE_GRANULARITY
					OnePhasePolicy;

			// Expansion kernel config
			typedef two_phase::expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				INSTRUMENT, 			// INSTRUMENT
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
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				7> 						// LOG_SCHEDULE_GRANULARITY
					ExpandPolicy;

			// Contraction kernel config
			typedef two_phase::contract_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				200,
				INSTRUMENT, 			// INSTRUMENT
				SATURATION_QUIT, 		// SATURATION_QUIT
				true, 					// DEQUEUE_PROBLEM_SIZE
				8,						// CTA_OCCUPANCY
				7,						// LOG_THREADS
				1,						// LOG_LOAD_VEC_SIZE
				2,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				0,						// BITMASK_CULL_THRESHOLD
				10> 					// LOG_SCHEDULE_GRANULARITY
					ContractPolicy;

			return EnactSearch<OnePhasePolicy, ExpandPolicy, ContractPolicy>(
				csr_problem, src, max_grid_size);
    	}

    	// GT200
    	if (cuda_props.device_sm_version >= 130) {

        	const int SATURATION_QUIT = 4 * 128;

        	// Single-grid tuning configuration
			typedef contract_expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,
				INSTRUMENT, 			// INSTRUMENT
				SATURATION_QUIT,		// SATURATION_QUIT
				1,						// CTA_OCCUPANCY
				8,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				1,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				-1,						// BITMASK_CULL_THRESHOLD
				6> 						// LOG_SCHEDULE_GRANULARITY
					OnePhasePolicy;

			// Expansion kernel config
			typedef two_phase::expand_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,
				INSTRUMENT, 			// INSTRUMENT
				1,						// CTA_OCCUPANCY
				8,						// LOG_THREADS
				0,						// LOG_LOAD_VEC_SIZE
				1,						// LOG_LOADS_PER_TILE
				5,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::ld::NONE,		// COLUMN_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_ALIGNED_READ_MODIFIER,
				util::io::ld::NONE,		// ROW_OFFSET_UNALIGNED_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				32,						// WARP_GATHER_THRESHOLD
				128 * 4, 				// CTA_GATHER_THRESHOLD,
				7> 						// LOG_SCHEDULE_GRANULARITY
					ExpandPolicy;

			// Contraction kernel config
			typedef two_phase::contract_atomic::KernelPolicy<
				typename CsrProblem::ProblemType,
				130,
				INSTRUMENT, 			// INSTRUMENT
				SATURATION_QUIT, 		// SATURATION_QUIT
				true, 					// DEQUEUE_PROBLEM_SIZE
				1,						// CTA_OCCUPANCY
				8,						// LOG_THREADS
				1,						// LOG_LOAD_VEC_SIZE
				1,						// LOG_LOADS_PER_TILE
				6,						// LOG_RAKING_THREADS
				util::io::ld::NONE,		// QUEUE_READ_MODIFIER,
				util::io::st::NONE,		// QUEUE_WRITE_MODIFIER,
				false,					// WORK_STEALING
				0,						// BITMASK_CULL_THRESHOLD
				10> 					// LOG_SCHEDULE_GRANULARITY
					ContractPolicy;

			return EnactSearch<OnePhasePolicy, ExpandPolicy, ContractPolicy>(
				csr_problem, src, max_grid_size);
	    }

		printf("Not yet tuned for this architecture\n");
		return cudaErrorInvalidDeviceFunction;
	}

};


} // namespace bfs
} // namespace graph
} // namespace b40c
