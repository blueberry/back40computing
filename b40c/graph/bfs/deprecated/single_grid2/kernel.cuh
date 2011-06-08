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

#include <b40c/bfs/compact_atomic/kernel.cuh>
#include <b40c/bfs/expand_atomic/kernel.cuh>

namespace b40c {
namespace bfs {
namespace hybrid {


/**
 * Kernel entry point
 */
template <typename KernelConfig, bool INSTRUMENT, int SATURATION_QUIT>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void Kernel(
	typename KernelConfig::VertexId 		src,
	typename KernelConfig::VertexId 		*d_in,
	typename KernelConfig::VertexId 		*d_parent_in,
	typename KernelConfig::VertexId 		*d_out,
	typename KernelConfig::VertexId 		*d_parent_out,
	typename KernelConfig::VertexId			*d_column_indices,
	typename KernelConfig::SizeT			*d_row_offsets,
	typename KernelConfig::VertexId			*d_source_path,
	typename KernelConfig::CollisionMask 	*d_collision_cache,
	util::CtaWorkProgress 					work_progress,
	util::GlobalBarrier						global_barrier,
	util::KernelRuntimeStats				kernel_stats,
	typename KernelConfig::VertexId			*d_iteration)
{
	typedef typename KernelConfig::ExpandConfig		ExpandConfig;
	typedef typename KernelConfig::CompactConfig	CompactConfig;
	typedef typename KernelConfig::VertexId 		VertexId;
	typedef typename KernelConfig::SizeT 			SizeT;

	// Shared storage for the kernel
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	if (INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStart();
	}

	if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

		// Reset all counters
		work_progress.template Reset<SizeT>();

		// Determine expand work decomposition for first iteration
		if (threadIdx.x == 0) {

			// Enqueue the source for us to subsequently process (we'll be the only
			// block with active work this iteration)
			util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(src, d_in);

			if (KernelConfig::MARK_PARENTS) {
				// Enqueue parent of source
				typename KernelConfig::VertexId parent = -2;
				util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(parent, d_parent_in);
			}

			// Obtain incoming queue size
			SizeT num_elements = 1;
			if (INSTRUMENT) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.expand.state.work_decomposition.template Init<ExpandConfig::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);
		}
	}

	// Barrier to protect work decomposition
	__syncthreads();

	VertexId iteration = 0;
	VertexId queue_index = 0;

	//---------------------------------------------------------------------
	// Expand
	//---------------------------------------------------------------------

	// Expand pass (don't do workstealing this iteration because without a
	// global barrier after queue-reset, the queue may be inconsistent
	// across CTAs)
	expand_atomic::SweepPass<ExpandConfig, false>::Invoke(
		iteration,
		queue_index,
		d_in,
		d_parent_in,
		d_out,
		d_parent_out,
		d_column_indices,
		d_row_offsets,
		d_source_path,
		work_progress,
		smem_storage.expand.state.work_decomposition,
		smem_storage.expand);

	iteration++;
	queue_index++;

	if (INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();
	}

	while (true) {

		// Global barrier
		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Compact
		//---------------------------------------------------------------------

		if (INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine compact work decomposition
		if (threadIdx.x == 0) {

			// Obtain incoming queue size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);

			// Initialize work decomposition in smem
			smem_storage.compact.state.work_decomposition.template Init<CompactConfig::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(queue_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Check if done
		if (!smem_storage.compact.state.work_decomposition.num_elements) {
			break;
		}

		// Compact pass
		compact_atomic::SweepPass<CompactConfig, CompactConfig::WORK_STEALING>::Invoke(
			queue_index,
			d_out,
			d_parent_out,
			d_in,
			d_parent_in,
			d_collision_cache,
			work_progress,
			smem_storage.compact.state.work_decomposition,
			smem_storage.compact);

		queue_index++;

		if (INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
		}

		// Global barrier
		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Expand
		//---------------------------------------------------------------------

		if (INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStart();
		}

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain incoming queue size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(queue_index);
			if (INSTRUMENT && (blockIdx.x == 0)) {
				kernel_stats.Aggregate(num_elements);
			}

			// Initialize work decomposition in smem
			smem_storage.expand.state.work_decomposition.template Init<ExpandConfig::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, queue_index + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(queue_index + 1);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Check if done
		if (!smem_storage.expand.state.work_decomposition.num_elements) {
			break;
		}

		// Expand pass
		expand_atomic::SweepPass<ExpandConfig, ExpandConfig::WORK_STEALING>::Invoke(
			iteration,
			queue_index,
			d_in,
			d_parent_in,
			d_out,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress,
			smem_storage.expand.state.work_decomposition,
			smem_storage.expand);

		iteration++;
		queue_index++;

		if (INSTRUMENT && (threadIdx.x == 0)) {
			kernel_stats.MarkStop();
		}
	}

	// Write out what iteration we stopped at
	if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
		d_iteration[0] = iteration;
	}

	// Flush stats
	if (INSTRUMENT && (threadIdx.x == 0)) {
		kernel_stats.MarkStop();	// We broke the while loop before marking stop
		kernel_stats.Flush();
	}
}

} // namespace compact_expand
} // namespace bfs
} // namespace b40c

