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

#include "cta_scan.cuh"
#include "../thread/thread_reduce.cuh"
#include "../thread/thread_scan.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


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
class CtaScan2D
{
	//---------------------------------------------------------------------
	// Type definitions and constants
	//---------------------------------------------------------------------

	/**
	 * Layout type for padded CTA raking grid
	 */
	typedef CtaScan<T, CTA_THREADS, POLICY> CtaScan;

public:

	/**
	 * Raking shared memory storage type
	 */
	typedef typename CtaScan::SmemStorage SmemStorage;

	//---------------------------------------------------------------------
	// Exclusive scan interface (with identity)
	//---------------------------------------------------------------------

	/**
	 * 2D exclusive CTA-wide prefix scan also producing aggregate.
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
		CtaScan::ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, aggregate);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * 2D exclusive CTA-wide prefix scan also producing aggregate and
	 * consuming/producing cta_prefix
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
		CtaScan::ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op, aggregate, cta_prefix);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * 2D exclusive CTA-wide prefix scan.
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
		CtaScan::ExclusiveScan(smem_storage, thread_partial, thread_partial, identity, scan_op);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	//---------------------------------------------------------------------
	// Exclusive sum interface
	//---------------------------------------------------------------------


	/**
	 * 2D exclusive CTA-wide prefix sum also producing aggregate.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		Sum<T> scan_op;
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveSum(smem_storage, thread_partial, thread_partial, aggregate);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


	/**
	 * 2D exclusive CTA-wide prefix sum also producing aggregate and
	 * consuming/producing cta_prefix.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		Sum<T> scan_op;
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveSum(smem_storage, thread_partial, thread_partial, aggregate, cta_prefix);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * 2D exclusive CTA-wide prefix sum.
	 */
	template <int ITEMS_PER_THREAD>						/// (inferred) The number of items per thread
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD])	/// (out) Output (may be aliased to input)
	{
		// Reduce consecutive thread items in registers
		Sum<T> scan_op;
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveSum(smem_storage, thread_partial, thread_partial);

		// Exclusive scan in registers with prefix
		ThreadScanExclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


	//---------------------------------------------------------------------
	// Inclusive scan interface
	//---------------------------------------------------------------------


	/**
	 * 2D inclusive CTA-wide prefix scan also producing aggregate
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
	 * 2D inclusive CTA-wide prefix scan also producing aggregate,
	 * consuming/producing cta_prefix.
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
	 * 2D inclusive CTA-wide prefix scan.
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


	//---------------------------------------------------------------------
	// Inclusive sum interface
	//---------------------------------------------------------------------

	/**
	 * 2D inclusive CTA-wide prefix sum also producing aggregate
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				&aggregate)						/// (out) Total aggregate (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		Sum<T> scan_op;
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveSum(smem_storage, thread_partial, thread_partial, scan_op, aggregate);

		// Inclusive scan in registers with prefix
		ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}


	/**
	 * 2D inclusive CTA-wide prefix sum also producing aggregate,
	 * consuming/producing cta_prefix.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD],	/// (out) Output (may be aliased to input)
		T				&aggregate,						/// (out) Total aggregate (valid in lane-0).
		T				&cta_prefix)					/// (in/out) Cta-wide prefix to scan (valid in lane-0).
	{
		// Reduce consecutive thread items in registers
		Sum<T> scan_op;
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveSum(smem_storage, thread_partial, thread_partial, scan_op, aggregate, cta_prefix);

		// Inclusive scan in registers with prefix
		ThreadScanInclusive(input, output, scan_op, thread_partial);
	}


	/**
	 * 2D inclusive CTA-wide prefix sum.
	 */
	template <
		int 			ITEMS_PER_THREAD,				/// (inferred) The number of items per thread
		typename 		ScanOp>							/// (inferred) Binary scan operator type
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage		&smem_storage,					/// (in) SmemStorage reference
		T 				(&input)[ITEMS_PER_THREAD],		/// (in) Input
		T 				(&output)[ITEMS_PER_THREAD])	/// (out) Output (may be aliased to input)
	{
		// Reduce consecutive thread items in registers
		Sum<T> scan_op;
		T thread_partial = ThreadReduce(input, scan_op);

		// Exclusive CTA-scan
		ExclusiveSum(smem_storage, thread_partial, thread_partial, scan_op);

		// Inclusive scan in registers with prefix
		ThreadScanInclusive(input, output, scan_op, thread_partial, (threadIdx.x != 0));
	}

};



} // namespace cub
CUB_NS_POSTFIX
