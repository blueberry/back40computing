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
class CtaScan
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

private:

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
	 *
	 */
	template <CtaScanPolicy	POLICY>
	struct InternalExclusiveScan;


	/**
	 * Specialized for raking scan
	 */
	template <>
	struct InternalExclusiveScan<CTA_SCAN_RAKING>
	{
		/**
		 * Raking warp-scan utility type
		 */
		typedef WarpScan<T, 1, RAKING_THREADS> WarpScan;

		/**
		 * Raking shared memory storage type
		 */
		struct SmemStorage
		{
			typename RakingWarpScan::SmemStorage 	warp_scan;		// Buffer for warp-synchronous scan
			typename CtaRakingGrid::SmemStorage 	raking_grid;	// Padded CTA raking grid
		};

	};


	/**
	 * Specialized for tiled warp scan
	 */
	template <>
	struct InternalExclusiveScan<CTA_SCAN_WARPSCANS>
	{
		/**
		 * Tiled warp-scan utility type
		 */
		typedef WarpScan<T, WARPS> TiledWarpScan;

		/**
		 * Tiled warpscan shared memory storage type
		 */
		typedef typename TiledWarpScan::SmemStorage SmemStorage;

	};



public:

	/**
	 * Shared memory storage type
	 */
	typedef If<(POLICY == CTA_SCAN_RAKING),
		RakingSmemStorage,
		TiledSmemStorage>::Type SmemStorage;


public:

	//---------------------------------------------------------------------
	// Exclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				identity,						/// (in) Identity value.
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		T thread_partial = ThreadReduce(input, scan_op);

		if (CTA_POLICY == CTA_SCAN_RAKING)
		{
			//
			// Raking scan with thread partials
			//

			// Place thread partial into shared memory raking grid
			T *placement_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
			*placement_ptr = thread_partial;

			__syncthreads();

			// Reduce parallelism down to just raking threads
			if (threadIdx.x < RAKING_THREADS)
			{
				// Raking upsweep reduction in grid
				T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);
				T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);

				// Warp synchronous scan
				WarpScan::ExclusiveScan(
					smem_storage.warp_scan,
					raking_partial,
					raking_partial,
					scan_op,
					identity,
					aggregate,
					cta_prefix);

				// Raking downsweep scan
				ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, raking_partial);
			}

			__syncthreads();

			// Grab thread prefix from shared memory
			thread_partial = *placement_ptr;
		}
		else
		{
			//
			// Tiled warp scan with thread partials
			//

			// Warp synchronous scan
			WarpScan::ExclusiveScan(
				smem_storage.warp_scan,
				raking_partial,
				exclusive_partial,
				scan_op,
				identity,
				aggregate,
				cta_prefix);

		}

		// Scan in registers, prefixed by the exclusive partial
		ThreadScanExclusive(
			input,
			output,
			scan_op,
			*placement_ptr);


	}


	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				identity,						/// (in) Identity value.
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		ExclusiveScan(smem_storage, input, output, scan_op, identity, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				identity)						/// (in) Identity value.
	{
		T aggregate;
		ExclusiveScan(smem_storage, input, output, scan_op, identity, aggregate);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				identity,						/// (in) Identity value.
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		typedef T Array1D[1];

		ExclusiveScan(
			smem_storage,
			reinterpret_cast<Array1D&>(input),
			reinterpret_cast<Array1D&>(output),
			scan_op,
			identity,
			aggregate,
			cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				identity,						/// (in) Identity value.
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		ExclusiveScan(smem_storage, input, output, scan_op, identity, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive scan over the input
	 * using the specified scan operator.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				identity)						/// (in) Identity value.
	{
		T aggregate;
		ExclusiveScan(smem_storage, input, output, scan_op, identity, aggregate);
	}


	//---------------------------------------------------------------------
	// Exclusive prefix sum interface
	//---------------------------------------------------------------------


	/**
	 * Perform a cooperative, CTA-wide exclusive prefix sum over the input.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		ExclusiveScan(smem_storage, input, output, Sum<T>(), T(0), aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive prefix sum over the input.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		ExclusiveSum(smem_storage, input, output, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive prefix sum over the input.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD])	/// (out) Output array (may be aliased to input)
	{
		T aggregate;
		ExclusiveSum(smem_storage, input, output, aggregate);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive prefix sum over the input.
	 */
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input array
		T 				&output,						/// (out) Output array (may be aliased to input)
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		typedef T Array1D[1];

		ExclusiveSum(
			smem_storage,
			reinterpret_cast<Array1D&>(input),
			reinterpret_cast<Array1D&>(output),
			aggregate,
			cta_prefix);
	}

	/**
	 * Perform a cooperative, CTA-wide exclusive prefix sum over the input.
	 */
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input array
		T 				&output,						/// (out) Output array (may be aliased to input)
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		ExclusiveSum(smem_storage, input, output, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide exclusive prefix sum over the input.
	 */
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input array
		T 				&output)						/// (out) Output array (may be aliased to input)
	{
		T aggregate;
		ExclusiveSum(smem_storage, input, output, aggregate);
	}

	//---------------------------------------------------------------------
	// Inclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Perform a cooperative, CTA-wide inclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{

		// Reduce in registers and place partial into shared memory raking grid
		T *placement_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
		*placement_ptr = ThreadReduce(input, scan_op);

		__syncthreads();

		// Reduce parallelism down to just raking threads
		if (threadIdx.x < RAKING_THREADS)
		{
			// Raking upsweep reduction in grid
			T *raking_ptr = CtaRakingGrid::RakingPtr(smem_storage.raking_grid);
			T raking_partial = ThreadReduce<RAKING_LENGTH>(raking_ptr, scan_op);

			// Warp synchronous scan
			T inclusive_partial;
			WarpScan::InclusiveScan(
				smem_storage.warp_scan,
				raking_partial,
				inclusive_partial,
				scan_op,
				aggregate,
				cta_prefix);

			// Raking downsweep scan
			ThreadScanExclusive<RAKING_LENGTH>(raking_ptr, raking_ptr, scan_op, inclusive_partial);
		}

		__syncthreads();

		// Scan in registers, prefixed by the inclusive partial from shared memory grid
		ThreadScanInclusive(
			input,
			output,
			scan_op,
			*placement_ptr,
			(threadIdx.x != 0));		// Thread0 does not apply prefix
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		InclusiveScan(smem_storage, input, output, scan_op, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive scan over the input
	 * using the specified scan operator.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		T aggregate;
		InclusiveScan(smem_storage, input, output, scan_op, aggregate);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive scan over the input
	 * using the specified scan operator.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		typedef T Array1D[1];

		InclusiveScan(
			smem_storage,
			reinterpret_cast<Array1D&>(input),
			reinterpret_cast<Array1D&>(output),
			scan_op,
			identity,
			aggregate,
			cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive scan over the input
	 * using the specified scan operator.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op,						/// (in) Binary scan operator
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		InclusiveScan(smem_storage, input, output, scan_op, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive scan over the input
	 * using the specified scan operator.
	 */
	template <typename ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input
		T 				&output,						/// (out) Output (may be aliased to input)
		ScanOp 			scan_op)						/// (in) Binary scan operator
	{
		T aggregate;
		InclusiveScan(smem_storage, input, output, scan_op, aggregate);
	}


	//---------------------------------------------------------------------
	// Inclusive prefix sum interface
	//---------------------------------------------------------------------


	/**
	 * Perform a cooperative, CTA-wide inclusive prefix sum over the input.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		InclusiveScan(smem_storage, input, output, Sum<T>(), T(0), aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive prefix sum over the input.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output array (may be aliased to input)
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		InclusiveSum(smem_storage, input, output, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive prefix sum over the input.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input array
		T 				(&output)[ITEMS_PER_THREAD])	/// (out) Output array (may be aliased to input)
	{
		T aggregate;
		InclusiveSum(smem_storage, input, output, aggregate);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive prefix sum over the input.
	 */
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input array
		T 				&output,						/// (out) Output array (may be aliased to input)
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		typedef T Array1D[1];

		InclusiveSum(
			smem_storage,
			reinterpret_cast<Array1D&>(input),
			reinterpret_cast<Array1D&>(output),
			aggregate,
			cta_prefix);
	}

	/**
	 * Perform a cooperative, CTA-wide inclusive prefix sum over the input.
	 */
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input array
		T 				&output,						/// (out) Output array (may be aliased to input)
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).  May be aliased with cta_prefix.
	{
		T cta_prefix;
		InclusiveSum(smem_storage, input, output, aggregate, cta_prefix);
	}


	/**
	 * Perform a cooperative, CTA-wide inclusive prefix sum over the input.
	 */
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				input,							/// (in) Input array
		T 				&output)						/// (out) Output array (may be aliased to input)
	{
		T aggregate;
		InclusiveSum(smem_storage, input, output, aggregate);
	}


};



} // namespace cub
CUB_NS_POSTFIX
