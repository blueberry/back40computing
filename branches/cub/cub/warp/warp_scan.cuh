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
struct WarpScan
{
	//---------------------------------------------------------------------
	// Constants and typedefs
	//---------------------------------------------------------------------

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

	// General iteration
	template <int COUNT, int MAX, bool HAS_IDENTITY>
	struct Iterate
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
			const int OFFSET = 1 << COUNT;

			// Share partial into buffer
			smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = partial;

			// Prevent compiler from hoisting variables between rounds
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


	//---------------------------------------------------------------------
	// Interface
	//---------------------------------------------------------------------


	/**
	 * Inclusive prefix sum.
	 */
	static __device__ __forceinline__ T InclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial) 			// Calling thread's input partial reduction
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Initialize identity region
		smem_storage.warp_scan[warp_id][lane_id] = 0;

		// Inclusive warp scan
		return Iterate<0, STEPS, true>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			partial,
			Sum<T>());
	}


	/**
	 * Inclusive prefix sum (with total aggregate).
	 */
	static __device__ __forceinline__ T InclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Inclusive warp scan
		partial = InclusiveSum(smem_storage, partial);

		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Share partial into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}


	/**
	 * Exclusive prefix sum.
	 */
	static __device__ __forceinline__ T ExclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial) 			// Calling thread's input partial reduction
	{
		// Compute exclusive warp scan from inclusive warp scan
		return InclusiveSum(smem_storage, partial) - partial;
	}


	/**
	 * Exclusive prefix sum (with total aggregate).
	 */
	static __device__ __forceinline__ T ExclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Inclusive warp scan
		partial = InclusiveSum(smem_storage, partial);

		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Share partial into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Retrieve exclusive scan
		partial = smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1];

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}


	/**
	 * Inclusive scan
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		ScanOp 			scan_op)			// Scan operator
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Inclusive warp scan
		return Iterate<0, STEPS, false>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			partial,
			scan_op);
	}


	/**
	 * Inclusive scan specialized for summation
	 */
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		Sum<T>)								// Scan operator
	{
		return InclusiveSum(smem_storage, partial);
	}


	/**
	 * Inclusive scan (with total aggregate)
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		ScanOp 			scan_op,			// Scan operator
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Inclusive warp scan
		partial = InclusiveScan(smem_storage, partial, scan_op);

		// Share partial into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Prevent compiler from hoisting variables between rounds
		if (!PRIMITIVE) __threadfence_block();

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}


	/**
	 * Inclusive scan specialized for summation (with total aggregate)
	 */
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		Sum<T>,								// Scan operator
		T				&aggregate)			// Total aggregate (out parameter)
	{
		return InclusiveSum(smem_storage, partial, aggregate);
	}


	/**
	 * Exclusive scan.
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		ScanOp 			scan_op,			// Scan operator
		T 				identity) 			// Identity element

	{
		// Warp, lane-IDs
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Initialize identity region
		smem_storage.warp_scan[warp_id][lane_id] = identity;

		// Inclusive warp scan
		partial = Iterate<0, STEPS, true>::InclusiveScan(
			smem_storage,
			warp_id,
			lane_id,
			partial,
			scan_op);

		// Share partial into buffer
		smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Prevent compiler from hoisting variables between rounds
		if (!PRIMITIVE) __threadfence_block();

		// Retrieve exclusive scan
		return smem_storage.warp_scan[warp_id][HALF_WARP_THREADS + lane_id - 1];
	}


	/**
	 * Exclusive scan specialized for summation
	 */
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		Sum<T>,								// Scan operator
		T) 									// Identity element

	{
		return ExclusiveSum(smem_storage, partial);
	}


	/**
	 * Exclusive scan (with total aggregate)
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		ScanOp 			scan_op,			// Scan operator
		T 				identity, 			// Identity element
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Exclusive warp scan
		partial = ExclusiveScan(smem_storage, partial, scan_op, identity);

		// Retrieve aggregate
		aggregate = smem_storage.warp_scan[warp_id][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}


	/**
	 * Exclusive scan specialized for summation (with total aggregate)
	 */
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		Sum<T>,								// Scan operator
		T,						 			// Identity element
		T				&aggregate)			// Total aggregate (out parameter)
	{
		return ExclusiveSum(smem_storage, partial, aggregate);
	}

};




} // namespace cub
CUB_NS_POSTFIX
