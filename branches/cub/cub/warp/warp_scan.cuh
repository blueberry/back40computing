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
 * Cooperative scan abstraction for warps.
 ******************************************************************************/

#pragma once

#include "../thread/thread_load.cuh"
#include "../thread/thread_store.cuh"
#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_wrapper.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Cooperative scan abstraction for warps.
 *
 * Performs a warp-synchronous Kogge-Stone style prefix scan.
 *
 * Features:
 * 		- Supports non-commutative scan operators.
 * 		- Supports concurrent scans within multiple warps.
 * 		- Supports logical warps smaller than the physical warp size (e.g., 8 threads)
 * 		- Zero bank conflicts for most types.
 *
 * Is most efficient when:
 * 		- ... the scan type T is a built-in primitive or CUDA vector type (e.g.,
 * 		  short, int2, double, float2, etc.)  Memory fences may be used
 * 		  to prevent reference reordering of non-primitive types.
 * 		- ... performing exclusive scans. Inclusive scans (other than prefix sum)
 *		  may use guarded memory accesses because no identity element is
 *		  provided.
 */
template <
	typename 	T,													// The reduction type
	int 		WARPS,												// The number of warps performing a warp scan
	int 		LOGICAL_WARP_THREADS = DeviceProps::WARP_THREADS>	// The number of threads per "warp" (may be less than the number of hardware warp threads)
class WarpScan
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

private:

	enum
	{
		// The number of warp scan steps
		STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE,

		// The number of threads in half a warp
		HALF_WARP_THREADS = 1 << (STEPS - 1),

		// The number of shared memory elements per warp
		WARP_SMEM_ELEMENTS =  LOGICAL_WARP_THREADS + HALF_WARP_THREADS,
	};

public:

	/**
	 * Shared memory storage type
	 */
	struct SmemStorage
	{
		T warp_scan[WARPS][WARP_SMEM_ELEMENTS];
	};


	//---------------------------------------------------------------------
	// Template iteration structures.  (Regular iteration cannot always be
	// unrolled due to conditionals or ABI procedure calls within
	// functors).
	//---------------------------------------------------------------------

private:

	// General template iteration
	template <int COUNT, int MAX, bool HAS_IDENTITY, bool SHARE_FINAL>
	struct Iterate
	{
		/**
		 * Inclusive scan step
		 */
		template <typename ScanOp>
		static __device__ __forceinline__ T InclusiveScan(
			SmemStorage 	&smem_storage,		// SmemStorage reference
			unsigned int 	warp_id,			// Warp id
			unsigned int 	lane_id,			// Lane id
			T 				partial, 			// Calling thread's input partial reduction
			ScanOp 			scan_op)			// Scan operator
		{
			const int OFFSET = 1 << COUNT;

			// Share partial into buffer
			ThreadStore<STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);

			// Update partial if addend is in range
			if (HAS_IDENTITY || (lane_id >= OFFSET))
			{
				T addend = ThreadLoad<LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - OFFSET]);

				partial = scan_op(partial, addend);
			}

			return Iterate<COUNT + 1, MAX, HAS_IDENTITY, SHARE_FINAL>::InclusiveScan(
				smem_storage,
				warp_id,
				lane_id,
				partial,
				scan_op);
		}
	};


	// Termination
	template <int MAX, bool HAS_IDENTITY, bool SHARE_FINAL>
	struct Iterate<MAX, MAX, HAS_IDENTITY, SHARE_FINAL>
	{
		// InclusiveScan
		template <typename ScanOp>
		static __device__ __forceinline__ T InclusiveScan(
			SmemStorage 	&smem_storage,		// SmemStorage reference
			unsigned int 	warp_id,			// Warp id
			unsigned int 	lane_id,			// Lane id
			T 				partial, 			// Calling thread's input partial reduction
			ScanOp 			scan_op)			// Scan operator
		{
			if (SHARE_FINAL)
			{
				// Share partial into buffer
				ThreadStore<STORE_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id], partial);
			}

			return partial;
		}
	};


public:

	//---------------------------------------------------------------------
	// Inclusive prefix sum interface
	//---------------------------------------------------------------------

	/**
	 * Inclusive prefix sum.
	 */
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output)			// (out) Calling thread's output.  May be aliased with input.
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Initialize identity region
		ThreadStore<STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

		// Compute inclusive warp scan (has identity, don't share final)
		output = Iterate<0, STEPS, true, false>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			input,
			Sum<T>());
	}


	/**
	 * Inclusive prefix sum.  Also computes warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Initialize identity region
		ThreadStore<STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], T(0));

		// Compute inclusive warp scan (has identity, share final)
		output = Iterate<0, STEPS, true, true>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			input,
			Sum<T>());

		// Retrieve aggregate in lane-0
		aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
	}


	/**
	 * Inclusive prefix sum, seeded by warp-wide prefix in lane-0.  Also computes
	 * warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void InclusiveSum(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		T				&aggregate,			// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)		// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		// Lane-IDs
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Incorporate warp-prefix from lane-0
		if (lane_id == 0)
		{
			input = warp_prefix + input;
		}

		// Compute inclusive warp scan
		InclusiveSum(smem_storage, input, output, aggregate);

		// Update warp_prefix
		warp_prefix += aggregate;
	}


	//---------------------------------------------------------------------
	// Exclusive prefix sum interface
	//---------------------------------------------------------------------

	/**
	 * Exclusive prefix sum.
	 */
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output)			// (out) Calling thread's output.  May be aliased with input.
	{
		// Compute exclusive warp scan from inclusive warp scan
		T inclusive;
		InclusiveSum(smem_storage, input, inclusive);
		output = inclusive - input;
	}


	/**
	 * Exclusive prefix sum.  Also computes warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		// Compute exclusive warp scan from inclusive warp scan
		T inclusive;
		InclusiveSum(smem_storage, input, inclusive, aggregate);
		output = inclusive - input;
	}


	/**
	 * Exclusive prefix sum, seeded by warp-wide prefix from lane-0.  Also computes
	 * warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void ExclusiveSum(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		T				&aggregate,			// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)		// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		// Warp, lane-IDs
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Incorporate warp-prefix from lane-0
		T partial = input;
		if (lane_id == 0)
		{
			partial = warp_prefix + input;
		}

		// Compute exclusive warp scan from inclusive warp scan
		T inclusive;
		InclusiveSum(smem_storage, partial, inclusive, aggregate);
		output = inclusive - input;

		// Update warp_prefix
		warp_prefix += aggregate;
	}


	//---------------------------------------------------------------------
	// Inclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Inclusive prefix scan.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op)			// (in) Scan operator.
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Compute inclusive warp scan (no identity, don't share final)
		output = Iterate<0, STEPS, false, false>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			input,
			scan_op);
	}


	/**
	 * Inclusive prefix scan (specialized for summation).
	 */
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		Sum<T, true>)						// (in) Scan operator.
	{
		InclusiveSum(smem_storage, input, output);
	}


	/**
	 * Inclusive prefix scan.  Also computes warp-wide aggregate in lane-0.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Compute inclusive warp scan (no identity, share final)
		output = Iterate<0, STEPS, false, true>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			input,
			scan_op);

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
	}


	/**
	 * Inclusive prefix scan (specialized for summation).  Also computes
	 * warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		Sum<T, true>,						// (in) Scan operator.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		InclusiveSum(smem_storage, input, output, aggregate);
	}


	/**
	 * Inclusive prefix scan, seeded by warp-wide prefix from lane-0.  Also computes
	 * warp-wide aggregate in lane-0.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				&aggregate,			// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)		// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		// Warp, lane-IDs
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Incorporate warp-prefix from lane-0
		if (lane_id == 0)
		{
			input = scan_op(warp_prefix, input);
		}

		// Compute inclusive warp scan
		InclusiveScan(smem_storage, input, output, scan_op, aggregate);

		// Update warp_prefix
		warp_prefix = scan_op(warp_prefix, aggregate);
	}


	/**
	 * Inclusive prefix scan (specialized for summation), seeded by warp-wide
	 * prefix from lane-0.  Also computes warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		Sum<T, true>,						// (in) Scan operator.
		T				&aggregate,			// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)		// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		InclusiveSum(smem_storage, input, output, aggregate, warp_prefix);
	}


	//---------------------------------------------------------------------
	// Exclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Exclusive prefix scan.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				identity)			// (in) Identity value.
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Initialize identity region
		ThreadStore<STORE_VS>(&smem_storage.warp_scan[warp_id][lane_id], identity);

		// Compute inclusive warp scan (identity, share final)
		T inclusive = Iterate<0, STEPS, true, true>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			input,
			scan_op);

		// Retrieve exclusive scan
		output = ThreadLoad<LOAD_VS>(&smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1]);
	}


	/**
	 * Exclusive prefix scan (specialized for summation).
	 */
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		Sum<T, true>,						// (in) Scan operator.
		T)									// (in) Identity value.
	{
		ExclusiveSum(smem_storage, input, output);
	}


	/**
	 * Exclusive prefix scan.  Also computes warp-wide aggregate in lane-0.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				identity,			// (in) Identity value.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x / LOGICAL_WARP_THREADS);

		// Exclusive warp scan
		ExclusiveScan(smem_storage, input, output, scan_op, identity);

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
	}


	/**
	 * Exclusive prefix scan (specialized for summation).  Also computes
	 * warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		Sum<T, true>,						// (in) Scan operator.
		T,									// (in) Identity value.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		ExclusiveSum(smem_storage, input, output, aggregate);
	}


	/**
	 * Exclusive prefix scan, seeded by warp-wide prefix from lane-0.  Also computes
	 * warp-wide aggregate in lane-0.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				identity,			// (in) Identity value.
		T				&aggregate,			// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)		// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		// Warp, lane-IDs
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (LOGICAL_WARP_THREADS - 1));

		// Incorporate warp-prefix from lane-0
		if (lane_id == 0)
		{
			input = scan_op(warp_prefix, input);
		}

		// Exclusive warp scan
		ExclusiveScan(smem_storage, input, output, scan_op, identity, aggregate);

		// Lane-0 gets warp_prefix (instead of identity)
		if (lane_id == 0)
		{
			output = warp_prefix;
			warp_prefix = scan_op(warp_prefix, aggregate);
		}
	}


	/**
	 * Exclusive prefix scan (specialized for summation), seeded by warp-wide
	 * prefix from lane-0.  Also computes warp-wide aggregate in lane-0.
	 */
	static __device__ __forceinline__ void ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		Sum<T, true>,						// (in) Scan operator.
		T,									// (in) Identity value.
		T				&aggregate,			// (out) Total aggregate (valid in lane-0).  May be aliased with warp_prefix.
		T				&warp_prefix)		// (in/out) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		ExclusiveSum(smem_storage, input, output, aggregate, warp_prefix);
	}

};


} // namespace cub
CUB_NS_POSTFIX
