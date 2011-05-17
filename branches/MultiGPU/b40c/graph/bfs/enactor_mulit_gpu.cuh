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
 * Multi-GPU compact-expand BFS implementation
 ******************************************************************************/

#pragma once

#include <vector>


#include <b40c/util/spine.cuh>
#include <b40c/util/kernel_runtime_stats.cuh>
#include <b40c/util/cta_work_progress.cuh>

#include <b40c/graph/bfs/enactor_base.cuh>
#include <b40c/graph/bfs/problem_type.cuh>

#include <b40c/graph/bfs/expand_atomic/kernel.cuh>
#include <b40c/graph/bfs/expand_atomic/kernel_policy.cuh>

#include <b40c/graph/bfs/partition_compact/policy.cuh>
#include <b40c/graph/bfs/partition_compact/upsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_compact/upsweep/kernel_policy.cuh>
#include <b40c/graph/bfs/partition_compact/downsweep/kernel.cuh>
#include <b40c/graph/bfs/partition_compact/downsweep/kernel_policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {



/**
 * Multi-GPU breadth-first-search enactor.
 *  
 * Each iterations is performed by its own kernel-launch.  
 */
class MultiGpuBfsEnactor : public BaseBfsEnactor
{

protected:

	static const int MAX_GPUS 			= 2;
	static const int RADIX_BITS 		= 1;

	typedef int SortingSpine[128 * (1 << RADIX_BITS) * 8];

	/**
	 * Temporary device storage needed for reducing partials produced
	 * by separate CTAs
	 */
	util::Spine spine[MAX_GPUS];

	// Queue size counters and accompanying functionality
	util::CtaWorkProgressLifetime work_progress[MAX_GPUS];

	/**
	 * Pinned memory for sorting spines
	 */
	volatile int (*sorting_spines)[128 * (1 << RADIX_BITS) * 8];
	int (*d_sorting_spines)[128 * (1 << RADIX_BITS) * 8];

public: 	
	
	/**
	 * Constructor
	 */
	MultiGpuBfsEnactor(
		bool DEBUG = false) :
			BaseBfsEnactor(DEBUG),
			sorting_spines(NULL)
	{
		// Allocate pinned sorting spines
		int flags = cudaHostAllocMapped;
		if (b40c::util::B40CPerror(cudaHostAlloc((void **)&sorting_spines, sizeof(SortingSpine) * MAX_GPUS, flags),
			"CsrGraph cudaHostAlloc row_offsets failed", __FILE__, __LINE__)) exit(1);

		// Map the pinned sorting spines into device pointers
		if (util::B40CPerror(cudaHostGetDevicePointer((void **)&d_sorting_spines, (void *) sorting_spines, 0),
			"LevelGridBfsEnactor cudaHostGetDevicePointer d_column_indices failed", __FILE__, __LINE__)) exit(1);
	}


	/**
	 * Destructor
	 */
	virtual ~MultiGpuBfsEnactor()
	{
		b40c::util::B40CPerror(cudaFreeHost((void *)sorting_spines),
			"MultiGpuBfsEnactor cudaFreeHost sorting_spines failed", __FILE__, __LINE__);
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
    	total_queued = 0;
    	search_depth = 0;
    	avg_live = 0;
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
		int 								max_grid_size = 0,
		int 								num_gpus = 0)
	{
		typedef typename CsrProblem::VertexId					VertexId;
		typedef typename CsrProblem::SizeT						SizeT;

		// Expansion kernel config
		typedef expand_atomic::SweepKernelConfig<
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
			6> ExpandAtomicSweep;


		// Compaction tuning configuration
		typedef compact::ProblemConfig<
			typename CsrProblem::ProblemType,
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



		// Radix sorting upsweep and downsweep configs
		typedef typename util::If<
			CsrProblem::ProblemType::MARK_PARENTS,
			VertexId,
			util::NullType>::Type									SortValueType;

		typedef sort_compact::ProblemConfig<
			VertexId,				// KeyType
			SortValueType,			// ValueType
			SizeT,					// SizeT

			// Common
			200,
			RADIX_BITS,				// RADIX_BITS: 					1-bit radix digits
			9,						// LOG_SCHEDULE_GRANULARITY: 	512 grain elements
			util::io::ld::NONE,		// CACHE_MODIFIER: 				Default (CA: cache all levels)
			util::io::st::NONE,		// CACHE_MODIFIER: 				Default (CA: cache all levels)
			false,					// EARLY_EXIT: 					No early termination if homogeneous digits

			// Upsweep Kernel
			8,						// UPSWEEP_CTA_OCCUPANCY: 		8 CTAs/SM
			7,						// UPSWEEP_LOG_THREADS: 		128 threads/CTA
			1,						// UPSWEEP_LOG_LOAD_VEC_SIZE: 	vec-2 loads
			0,						// UPSWEEP_LOG_LOADS_PER_TILE: 	1 loads/tile

			// Spine-scan Kernel
			1,										// SPINE_CTA_OCCUPANCY: 		Only 1 CTA/SM really needed
			8,										// SPINE_LOG_THREADS: 			128 threads/CTA
			2,										// SPINE_LOG_LOAD_VEC_SIZE:		vec-4 loads
			0,										// SPINE_LOG_LOADS_PER_TILE:	1 loads/tile
			5,

			// Downsweep Kernel
			4,						// DOWNSWEEP_CTA_OCCUPANCY: 		8 CTAs/SM
			8,						// DOWNSWEEP_LOG_THREADS: 			64 threads/CTA
			0,						// DOWNSWEEP_LOG_LOAD_VEC_SIZE: 	vec-4 loads
			1,						// DOWNSWEEP_LOG_LOADS_PER_CYCLE: 	2 loads/cycle
			0, 						// DOWNSWEEP_LOG_CYCLES_PER_TILE: 1 cycle/tile
			7>						// DOWNSWEEP_LOG_RAKING_THREADS: 4 warps
				SortCompactTuneConfig;

		// SortCompact upsweep kernel config
		typedef radix_sort::distribution::upsweep::KernelPolicy<
				typename SortCompactTuneConfig::Upsweep,
				util::NullType,
				0,
				0> SortUpsweep;

		// SortCompact spine kernel config
		typedef typename SortCompactTuneConfig::Spine SortSpine;

		// SortCompact downsweep kernel config
		typedef radix_sort::distribution::downsweep::KernelPolicy<
				typename SortCompactTuneConfig::Downsweep,
				util::NullType,
				util::NullType,
				0,
				0> SortDownsweep;






		cudaError_t retval = cudaSuccess;
		if (num_gpus == 0) {
			num_gpus = 1;
		}


		//
		// Determine grid size(s)
		//

		int expand_min_occupancy 	= ExpandAtomicSweep::CTA_OCCUPANCY;
		int expand_grid_size 		= MaxGridSize(expand_min_occupancy, max_grid_size);
		int sort_grid_size 			= expand_grid_size;

		if (DEBUG) printf("DEBUG: BFS expand min occupancy %d, level-grid size %d\n",
				expand_min_occupancy, expand_grid_size);

		int compact_min_occupancy 	= B40C_MIN((int) CompactUpsweep::CTA_OCCUPANCY, (int) CompactDownsweep::CTA_OCCUPANCY);
		int compact_grid_size 		= MaxGridSize(compact_min_occupancy, max_grid_size);
		int spine_elements 			= compact_grid_size;

		if (DEBUG) printf("DEBUG: BFS compact min occupancy %d, level-grid size %d\n",
			compact_min_occupancy, compact_grid_size);

		//
		// Configure our state
		//

		int bins_per_gpu = (1 << RADIX_BITS) / num_gpus;
		printf("Bins per gpu: %d\n", bins_per_gpu);

		// Configure sorting storage
		MultiCtaSortStorage<VertexId, SortValueType> 	sort_storage[MAX_GPUS];
		VertexId 										iteration[MAX_GPUS];
		VertexId										sub_iteration[MAX_GPUS];

		for (volatile int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			// Setup queue compaction spines
			if (retval = spine[gpu].template Setup<SizeT>(compact_grid_size, spine_elements)) exit(1);

			// Setup work progresses
			if (retval = work_progress[gpu].Setup()) exit(1);

			// Radix sorting storage
			sort_storage[gpu].num_elements 		= 0;
			sort_storage[gpu].d_keys[0] 		= bfs_problem.d_compact_queue[gpu];
			sort_storage[gpu].d_keys[1] 		= bfs_problem.d_expand_queue[gpu];
			sort_storage[gpu].d_values[0] 		= (SortValueType *) bfs_problem.d_compact_parent_queue[gpu];
			sort_storage[gpu].d_values[1] 		= (SortValueType *) bfs_problem.d_expand_parent_queue[gpu];

			iteration[gpu]						= 0;
			sub_iteration[gpu]					= 0;
		}


		// First iteration
		// Expand work queues
		for (volatile int gpu = 0; gpu < num_gpus; gpu++) {

			// Set device
			if (util::B40CPerror(cudaSetDevice(gpu),
				"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

			bool owns_source = (gpu == bfs_problem.Gpu(src));
			if (owns_source) {
				printf("GPU %d owns source %d\n", gpu, src);
			}

			expand_atomic::SweepKernel<ExpandAtomicSweep, INSTRUMENT, false>
					<<<expand_grid_size, ExpandAtomicSweep::THREADS, 0, bfs_problem.stream[gpu]>>>(
				(owns_source) ? src : -1,					// source
				(owns_source) ? 1 : 0,						// num_elements
				iteration[gpu],
				sub_iteration[gpu],
				num_gpus,
				NULL,
				sort_storage[gpu].d_keys[sort_storage[gpu].selector],							// sorted in
				(VertexId *) sort_storage[gpu].d_values[sort_storage[gpu].selector],			// sorted parents in
				sort_storage[gpu].d_keys[sort_storage[gpu].selector ^ 1],						// expanded out
				(VertexId *) sort_storage[gpu].d_values[sort_storage[gpu].selector ^ 1],		// expanded parents out
				bfs_problem.d_column_indices[gpu],
				bfs_problem.d_row_offsets[gpu],
				bfs_problem.d_source_path[gpu],
				work_progress[gpu]);
			if (DEBUG && util::B40CPerror(cudaThreadSynchronize(),
				"MultiGpuBfsEnactor SweepKernel failed", __FILE__, __LINE__)) exit(1);

			sort_storage[gpu].selector 		^= 1;
			iteration[gpu] 					+= 1;
			sub_iteration[gpu] 				+= 1;
		}

		// BFS passes
		while (true) {

			// This is an unnecessary synch point in that the host must read
			// the queue sizes in order to pass them to sorting
			bool done = true;
			for (volatile int gpu = 0; gpu < num_gpus; gpu++) {

				// Set device
				if (util::B40CPerror(cudaSetDevice(gpu),
					"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

				if (this->work_progress[gpu].GetQueueLength(iteration[gpu], sort_storage[gpu].num_elements)) exit(0);
				printf("Iteration %d GPU %d compact queued %d\n",
					iteration[gpu] - 1,
					gpu,
					sort_storage[gpu].num_elements);
				if (sort_storage[gpu].num_elements) {
					done = false;
				}

//				printf("Pre- sorting gpu %d:\n", gpu);
//				DisplayDeviceResults(
//					sort_storage[gpu].d_keys[sort_storage[gpu].selector],
//					sort_storage[gpu].num_elements);

			}
			if (done) {
				// All done in all GPUs
				break;
			}

			// Sort compacted queues on all GPUs
			for (volatile int gpu = 0; gpu < num_gpus; gpu++) {

				// Set device
				if (util::B40CPerror(cudaSetDevice(gpu),
					"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);


/*
				// Invoke upsweep sorting kernel
				sort_compact::upsweep::UpsweepKernel<SortUpsweep>
					<<<compact_grid_size, SortUpsweep::THREADS, 0, bfs_problem.stream[gpu]>>>(
						NULL,
						(SizeT *) d_sorting_spines[gpu],
						sort_storage[gpu].d_keys[work.problem_storage->selector],
						sort_storage[gpu].d_keys[work.problem_storage->selector ^ 1],
						work_progress[gpu]);

				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "LsbSortEnactor:: UpsweepKernel failed ", __FILE__, __LINE__))) break;
*/
				// Invoke spine scan
				scan::SpineKernel<SortSpine><<<1, SortSpine::THREADS, 0, bfs_problem.stream[gpu]>>>(
					(SizeT*) d_sorting_spines[gpu],
					(SizeT*) d_sorting_spines[gpu],
					spine_elements << RADIX_BITS);
/*
				if (DEBUG && (retval = util::B40CPerror(cudaThreadSynchronize(), "LsbSortEnactor:: SpineScanKernel failed ", __FILE__, __LINE__))) break;

				// Invoke downsweep sorting kernel
				sort_compact::downsweep::DownsweepKernel<SortDownsweep>
					<<<compact_grid_size, SortDownsweep::THREADS, 0, bfs_problem.stream[gpu]>>>(
						NULL,
						(SizeT *) d_sorting_spines[gpu],
						sort_storage[gpu].d_keys[work.problem_storage->selector],
						sort_storage[gpu].d_keys[work.problem_storage->selector ^ 1],
						sort_storage[gpu].d_values[work.problem_storage->selector],
						sort_storage[gpu].d_values[work.problem_storage->selector ^ 1],
						work_progress[gpu]);
*/


//				printf("Post sorting gpu %d:\n", gpu);
//				DisplayDeviceResults(
//					sort_storage[gpu].d_keys[sort_storage[gpu].selector],
//					sort_storage[gpu].num_elements);

			}

			// Synchronize all GPUs to protect spines
			for (volatile int gpu = 0; gpu < num_gpus; gpu++) {
				if (util::B40CPerror(cudaSetDevice(gpu),
					"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);
				if (util::B40CPerror(cudaDeviceSynchronize(),
					"MultiGpuBfsEnactor cudaDeviceSynchronize failed", __FILE__, __LINE__)) exit(1);
			}

			// Expand work queues
			for (volatile int gpu = 0; gpu < num_gpus; gpu++) {

				// Set device
				if (util::B40CPerror(cudaSetDevice(gpu),
					"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

				// Stream in and expand inputs from all gpus (including ourselves
				for (int i = 0; i < num_gpus; i++) {

					int peer_gpu 		= (gpu + i) % num_gpus;
					int queue_offset 	= sorting_spines[peer_gpu][bins_per_gpu * gpu * sort_grid_size];
					int queue_oob 		= sorting_spines[peer_gpu][bins_per_gpu * (gpu + 1) * sort_grid_size];
					int num_elements	= queue_oob - queue_offset;

//					printf("Gpu %d getting %d from gpu %d, queue_offset: %d @ %d, queue_oob: %d @ %d\n",
//						gpu, num_elements, peer_gpu,
//						queue_offset, bins_per_gpu * gpu * sort_grid_size,
//						queue_oob, bins_per_gpu * (gpu + 1) * sort_grid_size);
//					fflush(stdout);

					expand_atomic::SweepKernel<ExpandAtomicSweep, INSTRUMENT, false>
							<<<expand_grid_size, ExpandAtomicSweep::THREADS, 0, bfs_problem.stream[gpu]>>>(
						-1,
						num_elements,
						iteration[gpu],
						sub_iteration[gpu],
						num_gpus,
						NULL,
						sort_storage[peer_gpu].d_keys[sort_storage[peer_gpu].selector] + queue_offset,							// sorted in
						(VertexId *) sort_storage[peer_gpu].d_values[sort_storage[peer_gpu].selector] + queue_offset,			// sorted parents in
						sort_storage[gpu].d_keys[sort_storage[gpu].selector ^ 1],						// expanded out
						(VertexId *) sort_storage[gpu].d_values[sort_storage[gpu].selector ^ 1],		// expanded parents out
						bfs_problem.d_column_indices[gpu],
						bfs_problem.d_row_offsets[gpu],
						bfs_problem.d_source_path[gpu],
						work_progress[gpu]);
					if (DEBUG && util::B40CPerror(cudaThreadSynchronize(),
						"MultiGpuBfsEnactor SweepKernel failed", __FILE__, __LINE__)) exit(1);

					sub_iteration[gpu] += 1;
				}
			}

			// Synchronize all GPUs to so that they may read their peer expansion
			// queues without having the results be clobbered by run-ahead compaction
			for (volatile int gpu = 0; gpu < num_gpus; gpu++) {
				if (util::B40CPerror(cudaSetDevice(gpu),
					"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

				if (this->work_progress[gpu].GetQueueLength(iteration[gpu] + 1, sort_storage[gpu].num_elements)) exit(0);
				printf("Iteration %d GPU %d expand queued %d\n", iteration[gpu], gpu, sort_storage[gpu].num_elements);
			}

			// Compact work queues
			for (volatile int gpu = 0; gpu < num_gpus; gpu++) {

				// Set device
				if (util::B40CPerror(cudaSetDevice(gpu),
					"MultiGpuBfsEnactor cudaSetDevice failed", __FILE__, __LINE__)) exit(1);

				iteration[gpu]++;

				// Upsweep
				compact::UpsweepKernel<CompactUpsweep, INSTRUMENT>
						<<<compact_grid_size, CompactUpsweep::THREADS, 0, bfs_problem.stream[gpu]>>>(
					iteration[gpu],
					NULL,
					sort_storage[gpu].d_keys[sort_storage[gpu].selector ^ 1],			// expanded in
					bfs_problem.d_keep[gpu],
					(SizeT *) spine[gpu](),
					bfs_problem.d_collision_cache[gpu],
					work_progress[gpu]);
				if (DEBUG && util::B40CPerror(cudaThreadSynchronize(),
					"MultiGpuBfsEnactor UpsweepKernel failed", __FILE__, __LINE__)) exit(1);

				// Spine
				scan::SpineKernel<CompactSpine><<<1, CompactSpine::THREADS>>>(
					(SizeT*) spine[gpu](), (SizeT*) spine[gpu](), spine_elements);
				if (DEBUG && util::B40CPerror(cudaThreadSynchronize(),
					"MultiGpuBfsEnactor SpineKernel failed", __FILE__, __LINE__)) exit(1);

				// Downsweep
				compact::DownsweepKernel<CompactDownsweep, INSTRUMENT>
						<<<compact_grid_size, CompactDownsweep::THREADS, 0, bfs_problem.stream[gpu]>>>(
					iteration[gpu],
					sort_storage[gpu].d_keys[sort_storage[gpu].selector ^ 1],						// expanded in
					(VertexId *) sort_storage[gpu].d_values[sort_storage[gpu].selector ^ 1],		// expanded parents in
					bfs_problem.d_keep[gpu],
					sort_storage[gpu].d_keys[sort_storage[gpu].selector],							// compacted out
					(VertexId *) sort_storage[gpu].d_values[sort_storage[gpu].selector],			// compacted parents out
					(SizeT *) spine[gpu](),
					work_progress[gpu]);
				if (DEBUG && util::B40CPerror(cudaThreadSynchronize(),
					"MultiGpuBfsEnactor DownsweepKernel failed", __FILE__, __LINE__)) exit(1);
			}
		}

		return retval;
	}
    
};



} // namespace bfs
} // namespace graph
} // namespace b40c
