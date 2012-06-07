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

#include <cub/device_props.cuh>
#include <cub/type_utils.cuh>
#include <cub/operators.cuh>
#include <cub/ns_umbrella.cuh>

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
	typedef typename If<(CtaRakingGrid::PRIMITIVE), volatile T, T>::Type WarpT;


	/**
	 * Shared memory storage type
	 */
	typedef WarpT SmemStorage[WARPS][WARP_SMEM_ELEMENTS];


	//---------------------------------------------------------------------
	// Iteration structures
	//---------------------------------------------------------------------

	// General iteration
	template <int COUNT, int MAX, bool HAS_IDENTITY>
	struct Iterate
	{
		// InclusiveWarpScan
		template <typename ScanOp>
		static __device__ __forceinline__ T InclusiveWarpScan(
			SmemStorage 	&smem_storage,		// SmemStorage reference
			unsigned int 	warp_id,			// Warp id
			unsigned int 	lane_id,			// Lane id
			T 				partial, 			// Calling thread's input partial reduction
			ScanOp 			scan_op)			// Scan operator
		{
			const int OFFSET = 1 << COUNT;

			// Share partial into buffer
			smem_storage[warp_id][HALF_WARP_THREADS + lane_id] = partial;

			// Prevent compiler from hoisting variables between rounds
			if (!PRIMITIVE) __threadfence_block();

			// Update partial if addend is in range
			if (HAS_IDENTITY || (lane_id <= OFFSET))
			{
				T addend = smem_storage[warp_id][HALF_WARP_THREADS + lane_id - OFFSET];
				partial = scan_op(partial, addend);
			}

			return Iterate<COUNT + 1, MAX, UNGUARDED>::InclusiveWarpScan(
				smem_storage,
				warp_id,
				lane_id,
				partial,
				scan_op);
		}
	};

	// Termination
	template <int MAX, bool UNGUARDED>
	struct Iterate<MAX, MAX, UNGUARDED>
	{
		// InclusiveWarpScan
		template <typename ScanOp>
		static __device__ __forceinline__ T InclusiveWarpScan(
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
	 * Inclusive prefix sum
	 */
	static __device__ __forceinline__ T InclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial) 			// Calling thread's input partial reduction
	{
		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Lane id
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Initialize identity region
		smem_storage[warp_id][lane_id] = 0;

		// Inclusive warp scan
		return Iterate<0, STEPS, true>::InclusiveWarpScan(
			smem_storage,
			warp_id,
			lane_id,
			partial,
			Sum<T>());
	}


	/**
	 * Inclusive prefix sum
	 */
	static __device__ __forceinline__ T InclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial) 			// Calling thread's input partial reduction
	{
		// Inclusive warp scan
		partial = InclusiveSum(smem_storage, partial);

		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Share partial into buffer
		smem_storage[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Retrieve aggregate
		aggregate = smem_storage[warp][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}


	/**
	 * Exclusive prefix sum
	 */
	static __device__ __forceinline__ T ExclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial) 			// Calling thread's input partial reduction
	{
		// Compute exclusive warp scan from inclusive warp scan
		return InclusiveSum(smem_storage, partial) - partial;
	}


	/**
	 * Exclusive prefix sum
	 */
	static __device__ __forceinline__ T ExclusiveSum(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Inclusive warp scan
		partial = InclusiveSum(smem_storage, partial);

		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Share partial into buffer
		smem_storage[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Retrieve exclusive scan
		return smem_storage[warp_id][HALF_WARP_THREADS + lane_id - 1];

		// Retrieve aggregate
		aggregate = smem_storage[warp][WARP_SMEM_ELEMENTS - 1];

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
		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Lane id
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Inclusive warp scan
		return Iterate<0, STEPS, false>::Invoke(
			smem_storage,
			warp_id,
			lane_id,
			partial,
			scan_op);
	}


	/**
	 * Inclusive scan
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T InclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		ScanOp 			scan_op,			// Scan operator
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Inclusive warp scan
		partial = InclusiveScan(smem_storage, partial, scan_op);

		// Share partial into buffer
		smem_storage[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Prevent compiler from hoisting variables between rounds
		if (!PRIMITIVE) __threadfence_block();

		// Retrieve aggregate
		aggregate = smem_storage[warp][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}


	/**
	 * Exclusive scan
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		T 				identity, 			// Identity element
		ScanOp 			scan_op)			// Scan operator
	{
		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Lane id
		unsigned int lane_id = (WARPS == 1) ? threadIdx.x : (threadIdx.x & (DeviceProps::WARP_THREADS - 1));

		// Initialize identity region
		smem_storage[warp_id][lane_id] = identity;

		// Inclusive warp scan
		partial = Iterate<0, STEPS, true>::Invoke(
			smem_storage,
			warp_id,
			lane_id,
			partial,
			scan_op);

		// Share partial into buffer
		smem_storage[warp_id][HALF_WARP_THREADS + lane_id] = partial;

		// Prevent compiler from hoisting variables between rounds
		if (!PRIMITIVE) __threadfence_block();

		// Retrieve exclusive scan
		return smem_storage[warp_id][HALF_WARP_THREADS + lane_id - 1];
	}


	/**
	 * Exclusive scan
	 */
	template <typename ScanOp>
	static __device__ __forceinline__ T ExclusiveScan(
		SmemStorage 	&smem_storage,		// SmemStorage reference
		T 				partial, 			// Calling thread's input partial reduction
		T 				identity, 			// Identity element
		ScanOp 			scan_op,			// Scan operator
		T				&aggregate)			// Total aggregate (out parameter)
	{
		// Warp id
		unsigned int warp_id = (WARPS == 1) ? 0 : (threadIdx.x >> DeviceProps::LOG_WARP_THREADS);

		// Exclusive warp scan
		partial = ExclusiveScan(partial, scan_op, identity);

		// Retrieve aggregate
		aggregate = smem_storage[warp][WARP_SMEM_ELEMENTS - 1];

		return partial;
	}
};




} // namespace cub
CUB_NS_POSTFIX
