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
 * BFS atomic expansion kernel
 ******************************************************************************/

#pragma once

#include <b40c/util/cta_work_distribution.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/bfs/expand_atomic/sweep_cta.cuh>

namespace b40c {
namespace bfs {
namespace expand_atomic {


/**
 * Sweep expansion pass (non-workstealing)
 */
template <typename KernelConfig, bool WORK_STEALING>
struct SweepPass
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelConfig::VertexId 		&iteration,
		typename KernelConfig::VertexId 		&sub_iteration,
		typename KernelConfig::VertexId 		&num_gpus,
		typename KernelConfig::VertexId 		*&d_in,
		typename KernelConfig::VertexId 		*&d_parent_in,
		typename KernelConfig::VertexId 		*&d_out,
		typename KernelConfig::VertexId 		*&d_parent_out,
		typename KernelConfig::VertexId			*&d_column_indices,
		typename KernelConfig::SizeT			*&d_row_offsets,
		typename KernelConfig::VertexId			*&d_source_path,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
		SmemStorage								&smem_storage)
	{
		typedef SweepCta<KernelConfig, SmemStorage> 	SweepCta;
		typedef typename KernelConfig::SizeT 			SizeT;

		// Determine our threadblock's work range
		util::CtaWorkLimits<SizeT> work_limits;
		work_decomposition.template GetCtaWorkLimits<
			KernelConfig::LOG_TILE_ELEMENTS,
			KernelConfig::LOG_SCHEDULE_GRANULARITY>(work_limits);

		// Return if we have no work to do
		if (!work_limits.elements) {
			return;
		}

		// CTA processing abstraction
		SweepCta cta(
			iteration,
			num_gpus,
			smem_storage,
			d_in,
			d_parent_in,
			d_out,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress);

		// Process full tiles
		while (work_limits.offset < work_limits.guarded_offset) {

			cta.template ProcessTile<true>(work_limits.offset);
			work_limits.offset += KernelConfig::TILE_ELEMENTS;
		}

		// Clean up last partial tile with guarded-i/o
		if (work_limits.guarded_elements) {
			cta.template ProcessTile<false>(
				work_limits.offset,
				work_limits.out_of_bounds);
		}
	}
};


template <typename SizeT, typename IterationT>
__device__ __forceinline__ SizeT StealWork(
	util::CtaWorkProgress &work_progress,
	int count,
	IterationT sub_iteration)
{
	__shared__ SizeT s_offset;		// The offset at which this CTA performs tile processing, shared by all

	// Thread zero atomically steals work from the progress counter
	if (threadIdx.x == 0) {
		s_offset = work_progress.Steal<SizeT>(count, sub_iteration);
	}

	__syncthreads();		// Protect offset

	return s_offset;
}



/**
 * Sweep expansion pass (workstealing)
 */
template <typename KernelConfig>
struct SweepPass <KernelConfig, true>
{
	template <typename SmemStorage>
	static __device__ __forceinline__ void Invoke(
		typename KernelConfig::VertexId 		&iteration,
		typename KernelConfig::VertexId 		&sub_iteration,
		typename KernelConfig::VertexId 		&num_gpus,
		typename KernelConfig::VertexId 		*&d_in,
		typename KernelConfig::VertexId 		*&d_parent_in,
		typename KernelConfig::VertexId 		*&d_out,
		typename KernelConfig::VertexId 		*&d_parent_out,
		typename KernelConfig::VertexId			*&d_column_indices,
		typename KernelConfig::SizeT			*&d_row_offsets,
		typename KernelConfig::VertexId			*&d_source_path,
		util::CtaWorkProgress 					&work_progress,
		util::CtaWorkDistribution<typename KernelConfig::SizeT> &work_decomposition,
		SmemStorage								&smem_storage)
	{
		typedef SweepCta<KernelConfig, SmemStorage> 	SweepCta;
		typedef typename KernelConfig::SizeT 			SizeT;

		// CTA processing abstraction
		SweepCta cta(
			iteration,
			num_gpus,
			smem_storage,
			d_in,
			d_parent_in,
			d_out,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress);

		// Total number of elements in full tiles
		SizeT unguarded_elements = work_decomposition.num_elements & (~(KernelConfig::TILE_ELEMENTS - 1));

		// Worksteal full tiles, if any
		SizeT offset;
		while ((offset = StealWork<SizeT>(work_progress, KernelConfig::TILE_ELEMENTS, sub_iteration)) < unguarded_elements) {
			cta.template ProcessTile<true>(offset);
		}

		// Last CTA does any extra, guarded work (first tile seen)
		if (blockIdx.x == gridDim.x - 1) {
			cta.template ProcessTile<false>(unguarded_elements, work_decomposition.num_elements);
		}
	}
};


/******************************************************************************
 * Sweep Reduction Kernel Entrypoint
 ******************************************************************************/

/**
 * Sweep expansion kernel entry point
 */
template <typename KernelConfig, bool INSTRUMENT>
__launch_bounds__ (KernelConfig::THREADS, KernelConfig::CTA_OCCUPANCY)
__global__
void SweepKernel(
	typename KernelConfig::VertexId 		src,
	typename KernelConfig::SizeT			num_elements,
	typename KernelConfig::VertexId 		iteration,
	typename KernelConfig::VertexId 		sub_iteration,
	typename KernelConfig::VertexId			num_gpus,
	typename KernelConfig::VertexId 		*d_in,
	typename KernelConfig::VertexId 		*d_parent_in,
	typename KernelConfig::VertexId 		*d_out,
	typename KernelConfig::VertexId 		*d_parent_out,
	typename KernelConfig::VertexId			*d_column_indices,
	typename KernelConfig::SizeT			*d_row_offsets,
	typename KernelConfig::VertexId			*d_source_path,
	util::CtaWorkProgress 					work_progress,
	util::KernelRuntimeStats				kernel_stats = util::KernelRuntimeStats())
{
	typedef typename KernelConfig::SizeT SizeT;

	// Shared storage for the kernel
	__shared__ typename KernelConfig::SmemStorage smem_storage;

	if (INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStart();
		}
	}

	if (iteration == 0) {

		if (threadIdx.x < util::CtaWorkProgress::COUNTERS) {

			// Reset all counters
			work_progress.template Reset<SizeT>();

			// Determine work decomposition for first iteration
			if (threadIdx.x == 0) {

				if (src != -1) {

					// We'll be the only block on the only GPU with active work this iteration.
					// Enqueue the source for us to subsequently process.
					util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(src, d_in);

					if (KernelConfig::MARK_PARENTS) {
						// Enqueue parent of source
						typename KernelConfig::VertexId parent = -2;
						util::io::ModifiedStore<KernelConfig::QUEUE_WRITE_MODIFIER>::St(parent, d_parent_in);
					}
				}

				// Initialize work decomposition in smem
				smem_storage.work_decomposition.template Init<KernelConfig::LOG_SCHEDULE_GRANULARITY>(
					num_elements, gridDim.x);
			}
		}

		// Barrier to protect work decomposition
		__syncthreads();

		// Don't do workstealing this iteration because without a
		// global barrier after queue-reset, the queue may be inconsistent
		// across CTAs
		SweepPass<KernelConfig, false>::Invoke(
			iteration,
			sub_iteration,
			num_gpus,
			d_in,
			d_parent_in,
			d_out,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress,
			smem_storage.work_decomposition,
			smem_storage);

	} else {

		// Determine work decomposition
		if (threadIdx.x == 0) {

			// Initialize work decomposition in smem
			smem_storage.work_decomposition.template Init<KernelConfig::LOG_SCHEDULE_GRANULARITY>(
				num_elements, gridDim.x);

			// Reset our next outgoing queue counter to zero
			work_progress.template StoreQueueLength<SizeT>(0, iteration + 2);

			// Reset our next workstealing counter to zero
			work_progress.template PrepResetSteal<SizeT>(sub_iteration + 1);

		}

		// Barrier to protect work decomposition
		__syncthreads();

		SweepPass<KernelConfig, KernelConfig::WORK_STEALING>::Invoke(
			iteration,
			sub_iteration,
			num_gpus,
			d_in,
			d_parent_in,
			d_out,
			d_parent_out,
			d_column_indices,
			d_row_offsets,
			d_source_path,
			work_progress,
			smem_storage.work_decomposition,
			smem_storage);
	}

	if (INSTRUMENT) {
		if (threadIdx.x == 0) {
			kernel_stats.MarkStop();
		}
	}
}

} // namespace expand_atomic
} // namespace bfs
} // namespace b40c

