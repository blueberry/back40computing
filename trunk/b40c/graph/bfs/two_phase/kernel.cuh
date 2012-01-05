/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 * BFS atomic compact-expand kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>

#include <b40c/graph/bfs/compact_expand_atomic/cta.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace two_phase {


/******************************************************************************
 * Sweep Kernel Entrypoint
 ******************************************************************************/

/**
 * Sweep compact-expand kernel entry point
 */
template <typename KernelPolicy>
__launch_bounds__ (KernelPolicy::THREADS, KernelPolicy::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelPolicy::VertexId			iteration,
	typename KernelPolicy::VertexId			queue_index,
	typename KernelPolicy::VertexId			steal_index,
	typename KernelPolicy::VertexId 		src,
	typename KernelPolicy::VertexId 		*d_in,
	typename KernelPolicy::VertexId 		*d_out,
	typename KernelPolicy::VertexId 		*d_predecessor_in,
	typename KernelPolicy::VertexId 		*d_predecessor_out,

	typename KernelPolicy::VertexId			*d_column_indices,
	typename KernelPolicy::SizeT			*d_row_offsets,
	typename KernelPolicy::VertexId			*d_labels,
	typename KernelPolicy::VisitedMask 	*d_visited_mask,
	util::CtaWorkProgress 					work_progress,
	util::GlobalBarrier						global_barrier,

	util::KernelRuntimeStats				kernel_stats,
	typename KernelPolicy::VertexId			*d_iteration)
{
	typedef typename KernelPolicy::CompactKernelPolicy 	CompactKernelPolicy;
	typedef typename KernelPolicy::ExpandKernelPolicy 	ExpandKernelPolicy;
	typedef typename KernelPolicy::VertexId 			VertexId;
	typedef typename KernelPolicy::SizeT 				SizeT;

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
				util::io::ModifiedStore<ExpandKernelPolicy::QUEUE_WRITE_MODIFIER>::St(src, d_in);

				if (ExpandKernelPolicy::MARK_PREDECESSORS) {
					// Enqueue predecessor of source
					VertexId predecessor = -2;
					util::io::ModifiedStore<ExpandKernelPolicy::QUEUE_WRITE_MODIFIER>::St(predecessor, d_predecessor_in);
				}

				// Initialize work decomposition in smem
				SizeT num_elements = 1;
				smem_storage.compact.state.work_decomposition.template Init<CompactKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
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
			smem_storage.compact.state.work_decomposition.template Init<CompactKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
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
	compact_atomic::SweepPass<CompactKernelPolicy, false>::Invoke(
		iteration,
		queue_index,
		steal_index,
		num_gpus,
		d_in,
		d_out,
		d_predecessor_in,
		d_labels,
		d_visited_mask,
		work_progress,
		smem_storage.compact.state.work_decomposition,
		smem_storage.compact);

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

		expand_atomic::SweepPass<ExpandKernelPolicy, true>::Invoke(
//		expand_atomic::SweepPass<ExpandKernelPolicy, ExpandKernelPolicy::WORK_STEALING>::Invoke(
			queue_index,
			steal_index,
			num_gpus,
			d_out,
			d_in,
			d_predecessor_out,
			d_column_indices,
			d_row_offsets,
			work_progress,
			smem_storage.expand.state.work_decomposition,
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
			smem_storage.compact.state.work_decomposition.template Init<CompactKernelPolicy::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(steal_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Check if done
		if ((!smem_storage.compact.state.work_decomposition.num_elements) ||
			(ExpandKernelPolicy::SATURATION_QUIT && (smem_storage.compact.state.work_decomposition.num_elements > gridDim.x * ExpandKernelPolicy::TILE_ELEMENTS * ExpandKernelPolicy::SATURATION_QUIT)))
		{
			break;
		}

		compact_atomic::SweepPass<CompactKernelPolicy, false>::Invoke(
//		compact_atomic::SweepPass<CompactKernelPolicy, CompactKernelPolicy::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			steal_index,
			num_gpus,
			d_in,
			d_out,
			d_predecessor_in,
			d_labels,
			d_visited_mask,
			work_progress,
			smem_storage.compact.state.work_decomposition,
			smem_storage.compact);

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

