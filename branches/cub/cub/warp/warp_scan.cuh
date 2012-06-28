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

#include "../device_props.cuh"
#include "../type_utils.cuh"
#include "../operators.cuh"
#include "../ns_umbrella.cuh"

CUB_NS_PREFIX
namespace cub {


/**
 * Performs a warp-synchronous Kogge-Stone style prefix scan.
 */
template <
	int 		WARPS,			// The number of warps performing a warp scan
	typename 	T>				// The reduction type
class WarpScan
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

private:

	enum
	{
		// Whether or not the reduction type is a built-in primitive
		PRIMITIVE = NumericTraits<T>::PRIMITIVE,

		// The number of threads in half a warp
		HALF_WARP_THREADS = DeviceProps::WARP_THREADS / 2,

		// The number of shared memory elements per warp
		WARP_SMEM_ELEMENTS =  DeviceProps::WARP_THREADS + HALF_WARP_THREADS,

		// The number of warp scan steps
		STEPS = DeviceProps::LOG_WARP_THREADS,
	};


	/**
	 * Qualified type of T to use for warp-synchronous storage.  For
	 * built-in primitive types, we can use volatile qualifier (and can omit
	 * syncthreads when warp-synchronous)
	 */
	typedef typename If<PRIMITIVE, volatile T, T>::Type WarpT;

public:

	/**
	 * Shared memory storage type
	 */
	typedef struct SmemStorage
	{
		WarpT warp_scan[WARPS][WARP_SMEM_ELEMENTS];
	};


	//---------------------------------------------------------------------
	// Iteration structures
	//---------------------------------------------------------------------

private:

	// General template iteration
	template <int COUNT, int MAX, bool HAS_IDENTITY>
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
			smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = partial;

			// Prevent compiler from reordering or omitting memory accesses between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Update partial if addend is in range
			if (HAS_IDENTITY || (lane_id >= OFFSET))
			{
				T addend = smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - OFFSET];
				partial = scan_op(partial, addend);
			}

			return Iterate<COUNT + 1, MAX, HAS_IDENTITY>::InclusiveScan(
				smem_storage,
				warp_id,
				lane_id,
				partial,
				scan_op);
		}
	};

	// Termination
	template <int MAX, bool HAS_IDENTITY>
	struct Iterate<MAX, MAX, HAS_IDENTITY>
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
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Initialize identity region
		smem_storage.warp_scan[warp_id][lane_id] = 0;

		// Compute inclusive warp scan
		output = Iterate<0, STEPS, true>::InclusiveScan(
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
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Compute inclusive warp scan
		InclusiveSum(smem_storage, input, output);

		// Share partial into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = output;

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
		T				warp_prefix)		// (in) Warp-wide prefix to warp_prefix with (valid in lane-0)
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Incorporate warp-prefix from lane-0
		if (lane_id == 0)
		{
			input = warp_prefix + input;
		}

		// Compute inclusive warp scan
		InclusiveSum(smem_storage, input, output);

		// Share partial into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = output;

		// Retrieve aggregate in lane-0
		aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
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
		InclusiveSum(smem_storage, input, inclusive);
		output = inclusive - input;

		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Share inclusive into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = inclusive;

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];
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
		T				warp_prefix)		// (in) Warp-wide prefix to warp_prefix with (valid in lane-0)
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Incorporate warp-prefix from lane-0
		if (lane_id == 0)
		{
			input = warp_prefix + input;
		}

		// Compute exclusive warp scan from inclusive warp scan
		T inclusive;
		InclusiveSum(smem_storage, input, inclusive);
		output = inclusive - input;

		// Share inclusive into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = inclusive;

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];
	}


	//---------------------------------------------------------------------
	// Inclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Inclusive prefix scan.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op)			// (in) Scan operator.
	{
		if (Equals<ScanOp, Sum<T> >::VALUE)
		{
			// Specialized for summation
			InclusiveSum(smem_storage, input, output);
		}
		else
		{
			// Warp, lane-IDs
			unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
			unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

			// Compute inclusive warp scan (guarded because we have no identity element)
			output = Iterate<0, STEPS, false>::InclusiveScan(
				smem_storage,
				warp_id,
				lane_id,
				input,
				scan_op);
		}
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
		if (Equals<ScanOp, Sum<T> >::VALUE)
		{
			// Specialized for summation
			InclusiveSum(smem_storage, input, output, aggregate);
		}
		else
		{
			// Warp, lane-IDs
			unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
			unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

			// Compute inclusive warp scan
			output = InclusiveScan(smem_storage, partial, scan_op);

			// Share partial into buffer
			smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = output;

			// Prevent compiler from reordering or omitting memory accesses between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Retrieve aggregate
			aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];
		}
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
		T				warp_prefix)		// (in) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		if (Equals<ScanOp, Sum<T> >::VALUE)
		{
			// Specialized for summation
			InclusiveSum(smem_storage, input, output, aggregate, warp_prefix);
		}
		else
		{
			// Warp, lane-IDs
			unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
			unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

			// Incorporate warp-prefix from lane-0
			if (lane_id == 0)
			{
				input = warp_prefix + input;
			}

			// Compute inclusive warp scan
			output = InclusiveScan(smem_storage, partial, scan_op);

			// Share partial into buffer
			smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = output;

			// Prevent compiler from reordering or omitting memory accesses between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Retrieve aggregate
			aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];
		}
	}


	//---------------------------------------------------------------------
	// Inclusive prefix scan interface
	//---------------------------------------------------------------------

	/**
	 * Exclusive prefix scan.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				identity)			// (in) Identity value.
	{
		if (Equals<ScanOp, Sum<T> >::VALUE)
		{
			// Specialized for summation
			ExclusiveSum(smem_storage, input, output);
		}
		else
		{
			// Warp, lane-IDs
			unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
			unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

			// Initialize identity region
			smem_storage.warp_scan[warp_id][lane_id] = identity;

			// Inclusive warp scan
			T inclusive = Iterate<0, STEPS, true>::InclusiveScan(
				smem_storage,
				warp_id,
				lane_id,
				input,
				scan_op);

			// Share inclusive into buffer
			smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = inclusive;

			// Prevent compiler from reordering or omitting memory accesses between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Retrieve exclusive scan
			output = smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1];
		}
	}


	/**
	 * Exclusive prefix scan.  Also computes warp-wide aggregate in lane-0.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// (in) SmemStorage reference
		T 				input, 				// (in) Calling thread's input
		T				&output,			// (out) Calling thread's output.  May be aliased with input.
		ScanOp 			scan_op,			// (in) Scan operator.
		T				identity,			// (in) Identity value.
		T				&aggregate)			// (out) Total aggregate (valid in lane-0)
	{
		if (Equals<ScanOp, Sum<T> >::VALUE)
		{
			// Specialized for summation
			ExclusiveSum(smem_storage, input, output, aggregate);
		}
		else
		{
			// Warp id
			unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

			// Exclusive warp scan
			output = ExclusiveScan(smem_storage, partial, scan_op, identity);

			// Retrieve aggregate
			aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
		}
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
		T				warp_prefix)		// (in) Warp-wide prefix to warp_prefix with (valid in lane-0).
	{
		if (Equals<ScanOp, Sum<T> >::VALUE)
		{
			// Specialized for summation
			ExclusiveSum(smem_storage, input, output, aggregate, warp_prefix);
		}
		else
		{
			// Warp, lane-IDs
			unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
			unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

			// Initialize identity region
			smem_storage.warp_scan[warp_id][lane_id] = identity;

			// Prevent compiler from reordering or omitting memory accesses between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Incorporate warp-prefix from lane-0
			if (lane_id == 0)
			{
				smem_storage.warp_scan[warp_id][HALF_WARP_THREADS - 1] = warp_prefix;
			}

			// Inclusive warp scan
			T inclusive = Iterate<0, STEPS, true>::InclusiveScan(
				smem_storage,
				warp_id,
				lane_id,
				input,
				scan_op);

			// Share inclusive into buffer
			smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = inclusive;

			// Prevent compiler from reordering or omitting memory accesses between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Retrieve exclusive scan
			output = smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1];

			// Retrieve aggregate
			aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];
		}
	}

};




} // namespace cub
CUB_NS_POSTFIX
