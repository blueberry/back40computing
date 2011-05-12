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
 * BFS single-grid kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>

#include <b40c/bfs/expand_atomic/sweep_kernel.cuh>
#include <b40c/bfs/compact/upsweep_kernel.cuh>
#include <b40c/bfs/compact/downsweep_kernel.cuh>
#include <b40c/scan/spine_kernel.cuh>

namespace b40c {
namespace bfs {
namespace single_grid {


/**
 * Single grid kernel entry point
 */
template <typename ProblemConfig>
__launch_bounds__ (ProblemConfig::THREADS, ProblemConfig::CTA_OCCUPANCY)
__global__
void SweepKernel(
	typename ProblemConfig::VertexId 		src,

	typename ProblemConfig::VertexId 		*d_expand_queue,
	typename ProblemConfig::VertexId 		*d_parent_expand_queue,
	typename ProblemConfig::VertexId 		*d_compact_queue,
	typename ProblemConfig::VertexId 		*d_parent_compact_queue,

	typename ProblemConfig::VertexId		*d_column_indices,
	typename ProblemConfig::SizeT			*d_row_offsets,
	typename ProblemConfig::VertexId		*d_source_path,
	typename ProblemConfig::CollisionMask	*d_collision_cache,
	typename ProblemConfig::ValidFlag		*d_keep,
	typename ProblemConfig::SizeT			*d_spine,

	util::CtaWorkProgress 					work_progress,
	util::GlobalBarrier						global_barrier,
	int 									spine_elements)
{
	typedef typename ProblemConfig::ExpandSweep 		ExpandSweep;
	typedef typename ProblemConfig::CompactUpsweep 		CompactUpsweep;
	typedef typename ProblemConfig::CompactSpine 		CompactSpine;
	typedef typename ProblemConfig::CompactDownsweep 	CompactDownsweep;

	typedef typename ProblemConfig::VertexId 			VertexId;
	typedef typename ProblemConfig::SizeT 				SizeT;

	// Shared storage for the kernel
	__shared__ typename ProblemConfig::SmemStorage smem_storage;


	// Work management
	VertexId iteration = 0;

	if (blockIdx.x == 0) {

		// First iteration
		if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

			// Reset all counters
			work_progress.template Reset<SizeT>();

			// Setup queue
			if (threadIdx.x == 0) {

				// We'll be the only block with active work this iteration.
				// Enqueue the source for us to subsequently process.
				util::io::ModifiedStore<ExpandSweep::QUEUE_WRITE_MODIFIER>::St(src, d_compact_queue);

				if (ProblemConfig::MARK_PARENTS) {
					// Enqueue parent of source
					typename ProblemConfig::VertexId parent = -2;
					util::io::ModifiedStore<ExpandSweep::QUEUE_WRITE_MODIFIER>::St(parent, d_parent_compact_queue);
				}

				// Initialize work decomposition in smem
				SizeT num_elements = 1;
				smem_storage.expand_smem_storage.work_decomposition.template
					Init<ExpandSweep::LOG_SCHEDULE_GRANULARITY>(num_elements, gridDim.x);
			}
		}

		__syncthreads();

		// No workstealing this iteration
		expand_atomic::SweepPass<ExpandSweep, false>::Invoke(
			iteration,
			d_compact_queue,
			d_parent_compact_queue,
			d_expand_queue,
			d_parent_expand_queue,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress,
			smem_storage.expand_smem_storage.work_decomposition,
			smem_storage.expand_smem_storage);
	}

	iteration++;

	global_barrier.Sync();

	//---------------------------------------------------------------------
	// Upsweep
	//---------------------------------------------------------------------

	// Determine work decomposition
	if (threadIdx.x == 0) {

		// Obtain problem size
		SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(iteration);

		// Initialize work decomposition in smem
		smem_storage.upsweep_smem_storage.work_decomposition.template
			Init<CompactUpsweep::LOG_SCHEDULE_GRANULARITY>(num_elements, gridDim.x);
	}

	// Barrier to protect work decomposition
	__syncthreads();

	while (smem_storage.upsweep_smem_storage.work_decomposition.num_elements) {

		compact::UpsweepPass<CompactUpsweep>(
			d_expand_queue,
			d_keep,
			d_spine,
			d_collision_cache,
			smem_storage.upsweep_smem_storage.work_decomposition,
			smem_storage.upsweep_smem_storage);

		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Spine
		//---------------------------------------------------------------------

		scan::SpinePass<CompactSpine>(
			d_spine,
			d_spine,
			spine_elements,
			smem_storage.spine_smem_storage);


		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Downsweep
		//---------------------------------------------------------------------

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(iteration);

			// Initialize work decomposition in smem
			smem_storage.downsweep_smem_storage.work_decomposition.template
				Init<CompactDownsweep::LOG_SCHEDULE_GRANULARITY>(num_elements, gridDim.x);
		}

		// Barrier to protect work decomposition
		__syncthreads();

		compact::DownsweepPass<CompactDownsweep>(
			iteration,
			d_expand_queue,
			d_parent_expand_queue,
			d_keep,
			d_compact_queue,
			d_parent_compact_queue,
			d_spine,
			work_progress,
			smem_storage.downsweep_smem_storage.work_decomposition,
			smem_storage.downsweep_smem_storage);


		global_barrier.Sync();

		//---------------------------------------------------------------------
		// BFS expansion
		//---------------------------------------------------------------------

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(iteration);

			// Initialize work decomposition in smem
			smem_storage.expand_smem_storage.work_decomposition.template
				Init<ExpandSweep::LOG_SCHEDULE_GRANULARITY>(num_elements, gridDim.x);

			if (blockIdx.x == 0) {
				// Reset our next outgoing queue counter to zero
				work_progress.template StoreQueueLength<SizeT>(0, iteration + 2);

				if (ExpandSweep::WORK_STEALING) {
					// Reset our next workstealing counter to zero
					work_progress.template PrepResetSteal<SizeT>(iteration + 1);
				}
			}

		}

		// Barrier to protect work decomposition
		__syncthreads();

		expand_atomic::SweepPass<ExpandSweep, ExpandSweep::WORK_STEALING>::Invoke(
			iteration,
			d_compact_queue,
			d_parent_compact_queue,
			d_expand_queue,
			d_parent_expand_queue,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress,
			smem_storage.expand_smem_storage.work_decomposition,
			smem_storage.expand_smem_storage);

		iteration++;

		global_barrier.Sync();

		//---------------------------------------------------------------------
		// Upsweep
		//---------------------------------------------------------------------

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Obtain problem size
			SizeT num_elements = work_progress.template LoadQueueLength<SizeT>(iteration);

			// Initialize work decomposition in smem
			smem_storage.upsweep_smem_storage.work_decomposition.template
				Init<CompactUpsweep::LOG_SCHEDULE_GRANULARITY>(num_elements, gridDim.x);
		}

		// Barrier to protect work decomposition
		__syncthreads();
	}
}

} // namespace single_grid
} // namespace bfs
} // namespace b40c

