/******************************************************************************
 * 
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
 ******************************************************************************/

/******************************************************************************
 * BFS two-phase kernel (fused BFS iterations).
 *
 * Both contraction and expansion phases are fused within the same kernel,
 * separated by software global barriers.  The kernel itself also steps through
 * BFS iterations (without iterative kernel invocations) using software global
 * barriers.
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/two_phase/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/two_phase/contract_atomic/kernel.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace two_phase {


/******************************************************************************
 * Kernel entrypoint
 ******************************************************************************/

/**
 * Contract-expand kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId 		iteration,					// Current BFS iteration
	typename KernelPolicy::VertexId			queue_index,				// Current frontier queue counter index
	typename KernelPolicy::VertexId			steal_index,				// Current workstealing counter index
	typename KernelPolicy::VertexId 		src,						// Source vertex (may be -1 if iteration != 0)
	typename KernelPolicy::VertexId 		*d_edge_frontier,			// Edge frontier
	typename KernelPolicy::VertexId 		*d_vertex_frontier,			// Vertex frontier
	typename KernelPolicy::VertexId 		*d_predecessor,				// Predecessor edge frontier (used when KernelPolicy::MARK_PREDECESSORS)
	typename KernelPolicy::VertexId			*d_column_indices,			// CSR column-indices array
	typename KernelPolicy::SizeT			*d_row_offsets,				// CSR row-offsets array
	typename KernelPolicy::VertexId			*d_labels,					// BFS labels to set
	typename KernelPolicy::VisitedMask 		*d_visited_mask,			// Mask for detecting visited status
	util::CtaWorkProgress 					work_progress,				// Atomic workstealing and queueing counters
	typename KernelPolicy::SizeT			max_edge_frontier, 			// Maximum number of elements we can place into the outgoing edge frontier
	typename KernelPolicy::SizeT			max_vertex_frontier, 		// Maximum number of elements we can place into the outgoing vertex frontier
	util::GlobalBarrier						global_barrier,				// Software global barrier
	util::KernelRuntimeStats				kernel_stats,				// Kernel timing statistics (used when KernelPolicy::INSTRUMENT)
	typename KernelPolicy::VertexId			*d_iteration)				// Place to write final BFS iteration count
{
	typedef typename KernelPolicy::ContractKernelPolicy 	ContractKernelPolicy;
	typedef typename KernelPolicy::ExpandKernelPolicy 		ExpandKernelPolicy;
	typedef typename KernelPolicy::VertexId 				VertexId;
	typedef typename KernelPolicy::SizeT 					SizeT;

	int num_gpus = 1;

	// Shared storage for the kernel
	__shared__ typename KernelPolicy::SmemStorage smem_storage;

	if (KernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStart();
	}

	if (iteration == 0) {

		if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

			// Reset all counters
			work_progress.template Reset<SizeT>();

			// Determine work decomposition for first iteration
			if (threadIdx.x == 0) {

				// We'll be the only block with active work this iteration.
				// Enqueue the source for us to subsequently process.
				util::io::ModifiedStore<ExpandKernelPolicy::QUEUE_STORE_MODIFIER>::St(src, d_edge_frontier);

				if (ExpandKernelPolicy::MARK_PREDECESSORS) {
					// Enqueue predecessor of source
					VertexId predecessor = -2;
					util::io::ModifiedStore<ExpandKernelPolicy::QUEUE_STORE_MODIFIER>::St(predecessor, d_predecessor);
				}

				// Initialize work decomposition in smem
				SizeT num_elements = 1;
				smem_storage.contract.state.work_decomposition.template Init<ContractKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
					num_elements, gridDim.x);
			}
		}

	} else {

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (ExpandKernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.contract.state.work_decomposition.template Init<ContractKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

		}
	}

	// Barrier to protect work decomposition
	__syncthreads();

	// Don't do workstealing this iteration because without a
	// global barrier after queue-reset, the queue may be inconsistent
	// across CTAs
	contract_atomic::SweepPass<ContractKernelPolicy, false>::Invoke(
		iteration,
		queue_index,
		steal_index,
		num_gpus,
		d_edge_frontier,
		d_vertex_frontier,
		d_predecessor,
		d_labels,
		d_visited_mask,
		work_progress,
		smem_storage.contract.state.work_decomposition,
		max_vertex_frontier,
		smem_storage.contract);

	queue_index++;
	steal_index++;

	if (ExpandKernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
	}

	global_barrier.Sync();

	while (true) {

		//---------------------------------------------------------------------
		// Flip
		//---------------------------------------------------------------------

		if (ExpandKernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (ExpandKernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.expand.state.work_decomposition.template Init<ExpandKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);

		}

		// Barrier to protect work decomposition
		__syncthreads();

		if ((!smem_storage.expand.state.work_decomposition.num_elements) ||
			(ExpandKernelPolicy::SATURATION_QUIT && (smem_storage.expand.state.work_decomposition.num_elements > gridDim.x * ExpandKernelPolicy::TILE_ELEMENTS * ExpandKernelPolicy::SATURATION_QUIT)))
		{
			break;
		}

		expand_atomic::SweepPass<ExpandKernelPolicy, ExpandKernelPolicy::WORK_STEALING>::Invoke(
			queue_index,
			steal_index,
			num_gpus,
			d_vertex_frontier,
			d_edge_frontier,
			d_predecessor,
			d_column_indices,
			d_row_offsets,
			work_progress,
			smem_storage.expand.state.work_decomposition,
			max_edge_frontier,
			smem_storage.expand);

		iteration++;
		queue_index++;
		steal_index++;

		if (ExpandKernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
		}

		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Flop
		//---------------------------------------------------------------------

		if (ExpandKernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (ExpandKernelPolicy::INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.contract.state.work_decomposition.template Init<ContractKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Check if done
		if ((!smem_storage.contract.state.work_decomposition.num_elements) ||
			(ExpandKernelPolicy::SATURATION_QUIT && (smem_storage.contract.state.work_decomposition.num_elements > gridDim.x * ExpandKernelPolicy::TILE_ELEMENTS * ExpandKernelPolicy::SATURATION_QUIT)))
		{
			break;
		}

		contract_atomic::SweepPass<ContractKernelPolicy, ContractKernelPolicy::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			steal_index,
			num_gpus,
			d_edge_frontier,
			d_vertex_frontier,
			d_predecessor,
			d_labels,
			d_visited_mask,
			work_progress,
			smem_storage.contract.state.work_decomposition,
			max_vertex_frontier,
			smem_storage.contract);

		queue_index++;
		steal_index++;

		if (ExpandKernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
		}

		global_barrier.Sync();
	}

	// Write out our final iteration
	if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
		d_iteration[0] = iteration;
	}

	if (ExpandKernelPolicy::INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
		kernel_stats.Flush();
	}
}

} // namespace two_phase
} // namespace bfs
} // namespace graph
} // namespace b40c

