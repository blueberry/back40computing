/******************************************************************************
 * 
 * Copyright (c) 2010-2012, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2012, NVIDIA CORPORATION.  All rights reserved.
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
 ******************************************************************************/

/******************************************************************************
 * Cooperative scan abstraction for CTAs.
 ******************************************************************************/

#pragma once

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../warp/warp_scan.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Tuning policy for CTA-scans
 */
enum CtaScanPolicy
{
	CTA_SCAN_RAKING,		/// Use an work-efficient, but longer-latency algorithm (raking reduce-then-scan).  Useful when fully occupied.
	CTA_SCAN_WARPSCANS,		/// Use an work-inefficient, but shorter-latency algorithm (tiled warpscans).  Useful when under-occupied.
};



/**
 * Cooperative prefix scan abstraction for CTAs.
 *
 * Features:
 * 		- Very efficient (only two synchronization barriers).
 * 		- Zero bank conflicts for most types.
 * 		- Supports non-commutative scan operators.
 *
 * Is most efficient when:
 * 		- CTA_THREADS is a multiple of the warp size
 * 		- The scan type T is a built-in primitive type (int, float, double, etc.)
 */
template <
	typename 		T,							/// The data type to be scanned
	int 			CTA_THREADS,				/// The CTA size in threads
	CtaScanPolicy	POLICY = CTA_SCAN_RAKING>	/// (optional) CTA scan tuning policy
class CtaScan;


/**
 * Specialized for CTA_SCAN_RAKING
 */
template <
	typename 		T,							/// The data type to be scanned
	int 			CTA_THREADS>				/// The CTA size in threads
class CtaScan<T, CTA_THREADS, CTA_SCAN_RAKING>
{
private:

	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	/**
	 * Layout type for padded CTA raking grid
	 */
	typedef CtaRakingGrid<CTA_THREADS, T> CtaRakingGrid;

	enum
	{
		// Number of active warps
		WARPS = (CTA_THREADS + DeviceProps::WARP_THREADS - 1) / DeviceProps::WARP_THREADS,

		// Number of raking threads
		RAKING_THREADS = CtaRakingGrid::RAKING_THREADS,

		// Number of raking elements per warp synchronous raking thread
		RAKING_LENGTH = CtaRakingGrid::RAKING_LENGTH,

		// Cooperative work can be entirely warp synchronous
		WARP_SYNCHRONOUS = (CTA_THREADS == RAKING_THREADS),
	};

	/**
	 * Raking warp-scan utility type
	 */
	typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

public:

	/**
	 * Raking shared memory storage type
	 */
	struct SmemStorage
	{
		typename WarpScan::SmemStorage 			warp_scan;		// Buffer for warp-synchronous scan
		typename CtaRakingGrid::SmemStorage 	raking_grid;	// Padded CTA raking grid
	};

	//---------------------------------------------------------------------
	// Exclusive scan interface (with identity)
	//---------------------------------------------------------------------

	/**
	 * Exclusive CTA-wide prefix scan with aggregate
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		T				identity,						/// (in) Identity value.
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp scan
			WarpScan::ExclusiveScan(
				smem_storage.warp_scan,
				input,
				output,
				identity,
				scan_op,
				aggregate);
		}
		else
		{
			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
			*placement_ptr = input;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

				// Exclusive warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					identity,
					scan_op,
					aggregate);

				// Exclusive raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

				if (!CtaRakingGrid::UNGUARDED)
				{
					// CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
					aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
				}
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			output = *placement_ptr;

		}
	}


	/**
	 * Exclusive CTA-wide prefix scan with aggregate
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				identity,						/// (in) Identity value.
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, aggregate);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * Exclusive CTA-wide prefix scan with aggregate, with cta_prefix
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		T				identity,						/// (in) Identity value.
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp scan
			WarpScan::ExclusiveScan(
				smem_storage.warp_scan,
				input,
				output,
				identity,
				scan_op,
				aggregate,
				cta_prefix);
		}
		else
		{
			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
			*placement_ptr = input;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

				// Exclusvie warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					identity,
					scan_op,
					aggregate,
					cta_prefix);

				// Exclusive raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

				if (!CtaRakingGrid::UNGUARDED)
				{
					// CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
					aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
					cta_prefix = scan_op(cta_prefix, aggregate);
				}
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			output = *placement_ptr;
		}
	}


	/**
	 * Exclusive CTA-wide prefix scan with aggregate, with cta_prefix
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				identity,						/// (in) Identity value.
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, aggregate, cta_prefix);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * Exclusive CTA-wide prefix scan.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		T				identity,						/// (in) Identity value.
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		T aggregate;
		ExclusiveScan(smem_storage, input, output, identity, scan_op, aggregate);
	}



	/**
	 * Exclusive CTA-wide prefix scan.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				identity,						/// (in) Identity value.
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	//---------------------------------------------------------------------
	// Exclusive scan interface (without identity)
	//---------------------------------------------------------------------

	/**
	 * Exclusive CTA-wide prefix scan with aggregate, without identity (the
	 * output computed for thread-0 is invalid)
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp scan
			WarpScan::ExclusiveScan(
				smem_storage.warp_scan,
				input,
				output,
				scan_op,
				aggregate);
		}
		else
		{
			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
			*placement_ptr = input;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

				// Exclusive warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					scan_op,
					aggregate);

				// Exclusive raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));

				if (!CtaRakingGrid::UNGUARDED)
				{
					// CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
					aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
				}
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			output = *placement_ptr;
		}
	}


	/**
	 * Exclusive CTA-wide prefix scan with aggregate, without identity (the
	 * first output element computed for thread-0 is invalid)
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


	/**
	 * Exclusive CTA-wide prefix scan with aggregate, with cta_prefix, without
	 * identity (the output computed for thread-0 is invalid)
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp scan
			WarpScan::ExclusiveScan(
				smem_storage.warp_scan,
				input,
				output,
				scan_op,
				aggregate,
				cta_prefix);
		}
		else
		{
			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
			*placement_ptr = input;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

				// Exclusive warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					scan_op,
					aggregate,
					cta_prefix);

				// Exclusive raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

				if (!CtaRakingGrid::UNGUARDED)
				{
					// CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
					aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
					cta_prefix = scan_op(cta_prefix, aggregate);
				}
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			output = *placement_ptr;
		}
	}


	/**
	 * Exclusive CTA-wide prefix scan with aggregate, with cta_prefix, without
	 * identity (the first output element computed for thread-0 is invalid)
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate, cta_prefix);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * Exclusive CTA-wide prefix scan without identity (the output computed for
	 * thread-0 is invalid).
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		T aggregate;
		ExclusiveScan(smem_storage, input, output, scan_op, aggregate);
	}


	/**
	 * Exclusive CTA-wide prefix scan without identity (the first output element computed for
	 * thread-0 is invalid).
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


	//---------------------------------------------------------------------
	// Inclusive scan interface
	//---------------------------------------------------------------------

	/**
	 * Inclusive CTA-wide prefix scan with aggregate
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp scan
			WarpScan::InclusiveScan(
				smem_storage.warp_scan,
				input,
				output,
				scan_op,
				aggregate);
		}
		else
		{
			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
			*placement_ptr = input;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

				// Exclusive warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					scan_op,
					aggregate);

				// Exclusive raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial, (threadIdx.x != 0));

				if (!CtaRakingGrid::UNGUARDED)
				{
					// CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
					aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
				}
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			output = *placement_ptr;
		}
	}


	/**
	 * Inclusive CTA-wide prefix scan with aggregate
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate);

		// Inclusive scan in registers with prefix
		ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


	/**
	 * Inclusive CTA-wide prefix scan with aggregate, with cta_prefix, without
	 * identity (the output computed for thread-0 is invalid)
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		if (WARP_SYNCHRONOUS)
		{
			// Short-circuit directly to warp scan
			WarpScan::InclusiveScan(
				smem_storage.warp_scan,
				input,
				output,
				scan_op,
				aggregate,
				cta_prefix);
		}
		else
		{
			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::PlacementPtr(smem_storage.raking_grid);
			*placement_ptr = input;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

				// Warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					scan_op,
					aggregate,
					cta_prefix);

				// Exclusive raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);

				if (!CtaRakingGrid::UNGUARDED)
				{
					// CTA size isn't a multiple of warp size, so grab aggregate from the appropriate raking cell
					aggregate = *CtaRakingGrid::PlacementPtr(smem_storage.raking_grid, 0, CTA_THREADS);
					cta_prefix = scan_op(cta_prefix, aggregate);
				}
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			output = *placement_ptr;
		}
	}


	/**
	 * Inclusive CTA-wide prefix scan with aggregate, with cta_prefix, without
	 * identity (the first output element computed for thread-0 is invalid)
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op, aggregate, cta_prefix);

		// Inclusive scan in registers with prefix
		ThreadScanInclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * Inclusive CTA-wide prefix scan without identity (the output computed for
	 * thread-0 is invalid).
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		T aggregate;
		InclusiveScan(smem_storage, input, output, scan_op);
	}


	/**
	 * Inclusive CTA-wide prefix scan without identity (the first output element computed for
	 * thread-0 is invalid).
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveScan(smem_storage, thread_partial, thread_partial, scan_op);

		// Inclusive scan in registers with prefix
		ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


};



} // namespace cub
CUB_NS_POSTFIX
