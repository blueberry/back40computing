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
#include <b40c/bfs/compact/problem_config.cuh>
#include <b40c/bfs/expand_atomic/sweep_kernel.cuh>
#include <b40c/bfs/expand_atomic/sweep_kernel_config.cuh>
#include <b40c/bfs/compact/upsweep_kernel.cuh>
#include <b40c/bfs/compact/downsweep_kernel.cuh>

#include <b40c/scan/spine_kernel.cuh>

namespace b40c {
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
	 * Temporary device storage needed for reducing partials produced
	 * by separate CTAs
	 */
	util::Spine spine;

	/**
	 * CTA duty kernel stats
	 */
	util::KernelRuntimeStatsLifetime expand_kernel_stats;
	util::KernelRuntimeStatsLifetime compact_kernel_stats;
	long long total_avg_live;
	long long total_max_live;

	volatile long long *iteration;
	volatile long long *total_queued;

	long long *d_iteration;
	long long *d_total_queued;


public: 	
	
	/**
	 * Constructor
	 */
	LevelGridBfsEnactor(bool DEBUG = false) :
		BaseBfsEnactor(DEBUG),
		iteration(NULL),
		total_queued(NULL),
		d_iteration(NULL),
		d_total_queued(NULL),
		total_avg_live(0),
		total_max_live(0)
	{
		int flags = cudaHostAllocMapped;

		// Allocate pinned memory
		if (util::B40CPerror(cudaHostAlloc((void **)&iteration, sizeof(long long) * 1, flags),
			"LevelGridBfsEnactor cudaHostAlloc iteration failed", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaHostAlloc((void **)&total_queued, sizeof(long long) * 1, flags),
			"LevelGridBfsEnactor cudaHostAlloc total_queued failed", __FILE__, __LINE__)) exit(1);

		// Map into GPU space
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_iteration, (void *) iteration, 0),
			"LevelGridBfsEnactor cudaHostGetDevicePointer iteration failed", __FILE__, __LINE__)) exit(1);
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_total_queued, (void *) total_queued, 0),
			"LevelGridBfsEnactor cudaHostGetDevicePointer total_queued failed", __FILE__, __LINE__)) exit(1);

		total_queued[0] = 0;
	}


	/**
	 * Destructor
	 */
	virtual ~LevelGridBfsEnactor()
	{
		if (iteration) util::B40CPerror(cudaFreeHost((void *) iteration), "LevelGridBfsEnactor cudaFreeHost iteration failed", __FILE__, __LINE__);
		if (total_queued) util::B40CPerror(cudaFreeHost((void *) total_queued), "LevelGridBfsEnactor cudaFreeHost total_queued failed", __FILE__, __LINE__);
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
    	total_queued = this->total_queued[0];
    	search_depth = iteration[0] - 1;
    	avg_live = double(total_avg_live) / total_max_live;
    }
    
	/**
	 * Enacts a breadth-first-search on the specified graph problem.
	 *
	 * @return cudaSuccess on success, error enumeration otherwise
	 */
    template <bool INSTRUMENT, typename BfsCsrProblem>
	cudaError_t EnactSearch(
		BfsCsrProblem 						&bfs_problem,
		typename BfsCsrProblem::VertexId 	src,
		int 								max_grid_size = 0)
	{
		// Expansion kernel config
		typedef expand_atomic::SweepKernelConfig<
			typename BfsCsrProblem::ProblemType,
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
			6> ExpandAtomicSweep;


		// Compaction tuning configuration
		typedef compact::ProblemConfig<
			typename BfsCsrProblem::ProblemType,
			200,
			util::io::ld::NONE,
			util::io::st::NONE,
			9,

			// Compact upsweep
			8,
			7,
			0,
			0,

			// Compact spine
			5,
			2,
			0,
			5,

			// Compact downsweep
			8,
			7,
			1,
			1,
			5> CompactProblemConfig;


		typedef typename CompactProblemConfig::CompactUpsweep 		CompactUpsweep;
		typedef typename CompactProblemConfig::CompactSpine 		CompactSpine;
		typedef typename CompactProblemConfig::CompactDownsweep 	CompactDownsweep;

		typedef typename BfsCsrProblem::VertexId					VertexId;
		typedef typename BfsCsrProblem::SizeT						SizeT;

		//
		// Determine grid size(s)
		//

		int expand_min_occupancy = ExpandAtomicSweep::CTA_OCCUPANCY;
		int expand_grid_size = MaxGridSize(expand_min_occupancy, max_grid_size);

		printf("DEBUG: BFS expand min occupancy %d, level-grid size %d\n",
				expand_min_occupancy, expand_grid_size);

		int compact_min_occupancy = B40C_MIN((int) CompactUpsweep::CTA_OCCUPANCY, (int) CompactDownsweep::CTA_OCCUPANCY);
		int compact_grid_size = MaxGridSize(compact_min_occupancy, max_grid_size);

		printf("DEBUG: BFS compact min occupancy %d, level-grid size %d\n",
			compact_min_occupancy, compact_grid_size);



		cudaError_t retval = cudaSuccess;

		// Make sure our spine is big enough
		int spine_elements = compact_grid_size;
		if (retval = spine.Setup<SizeT>(compact_grid_size, spine_elements)) exit(1);

		// Make sure our runtime stats are good
		if (retval = expand_kernel_stats.Setup(compact_grid_size)) exit(1);
		if (retval = compact_kernel_stats.Setup(compact_grid_size)) exit(1);


		iteration[0] = 0;
		SizeT expand_queue_length;

		while (true) {

			// BFS iteration
			expand_atomic::SweepKernel<ExpandAtomicSweep, INSTRUMENT><<<expand_grid_size, ExpandAtomicSweep::THREADS>>>(
				src,
				iteration[0],
				bfs_problem.d_compact_queue,			// in
				bfs_problem.d_compact_parent_queue,
				bfs_problem.d_expand_queue,				// out
				bfs_problem.d_expand_parent_queue,
				bfs_problem.d_column_indices,
				bfs_problem.d_row_offsets,
				bfs_problem.d_source_path,
				this->work_progress,
				this->expand_kernel_stats);

			if (INSTRUMENT) {
				// expand barrier wait
				expand_kernel_stats.Accumulate(expand_grid_size, total_avg_live, total_max_live);
			}

			iteration[0]++;

			if (this->work_progress.GetQueueLength(iteration[0], expand_queue_length)) exit(0);
			total_queued[0] += expand_queue_length;
//			printf("%lld, %lld\n", iteration[0], (long long) queue_length);
			if (!expand_queue_length) {
				break;
			}

			// Upsweep
			compact::UpsweepKernel<CompactUpsweep, INSTRUMENT><<<compact_grid_size, CompactUpsweep::THREADS>>>(
				iteration[0],
				bfs_problem.d_expand_queue,				// in
				bfs_problem.d_keep,
				(SizeT *) this->spine(),
				bfs_problem.d_collision_cache,
				this->work_progress,
				this->compact_kernel_stats);

			// Get compact upsweep barrier duty %
			if (INSTRUMENT) {
				compact_kernel_stats.Accumulate(compact_grid_size, total_avg_live, total_max_live);
			}

			// Spine
			scan::SpineKernel<CompactSpine><<<1, CompactSpine::THREADS>>>(
				(SizeT*) spine(), (SizeT*) spine(), spine_elements);

			// Downsweep
			compact::DownsweepKernel<CompactDownsweep, INSTRUMENT><<<compact_grid_size, CompactDownsweep::THREADS>>>(
				iteration[0],
				bfs_problem.d_expand_queue,				// in
				bfs_problem.d_expand_parent_queue,
				bfs_problem.d_keep,
				bfs_problem.d_compact_queue,			// out
				bfs_problem.d_compact_parent_queue,
				(SizeT *) this->spine(),
				this->work_progress,
				this->compact_kernel_stats);

			if (INSTRUMENT) {
				// Get compact downsweep barrier duty %
				compact_kernel_stats.Accumulate(compact_grid_size, total_avg_live, total_max_live);
			}
		}

		return retval;
	}
    
};




} // namespace bfs
} // namespace b40c

