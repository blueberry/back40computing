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
 * Scan over thread-local array types
 ******************************************************************************/

#pragma once

#include "../operators.cuh"
#include "../ptx_intrinsics.cuh"
#include "../type_utils.cuh"
#include "../ns_umbrella.cuh"

CUB_NS_PREFIX
namespace cub {


/******************************************************************************
 * Exclusive scan
 ******************************************************************************/

/**
 * Computes an exclusive scan across the input array, storing the results into
 * the specified output array.  Returns the aggregate.
 */
template <
	int LENGTH,				// Length of input/output arrays
	typename T,				// Input/output type
	typename ScanOp>		// Binary scan operator type (parameters of type T)
__device__ __forceinline__ T ThreadScanExclusive(
	T			*input,		// Input array
	T			*output,	// Output array (may be aliased to input)
	ScanOp 		scan_op,
	T 			seed)
{
	T exclusive_partial = seed;
	T inclusive_partial = seed;

	#pragma unroll
	for (int i = 0; i < LENGTH; ++i)
	{
		inclusive_partial = scan_op(exclusive_partial, input[i]);
		output[i] = (T) exclusive_partial;
		exclusive_partial = inclusive_partial;
	}

	return inclusive_partial;
}


/**
 * Serial exclusive scan with the specified operator and seed
 */
template <
	int LENGTH,				// Length of input/output arrays
	typename T,				// Source type
	typename T,				// Target type
	typename ScanOp>		// Binary scan operator type (parameters of type T)
__device__ __forceinline__ T ThreadScanExclusive(
	T			(&input)[LENGTH],		// Input array
	T			(&output)[LENGTH],		// Output array (may be aliased to input)
	ScanOp 		scan_op,
	T 			seed)
{
	return ThreadScanExclusive<LENGTH>((T*) input, (T*) output, scan_op, seed);
}


/**
 * Serial exclusive scan with the addition operator and seed
 */
template <
	int LENGTH,
	typename T>
__device__ __forceinline__ T ThreadSumExclusive(
	T			*data,
	T 			seed)
{
	Sum<T> scan_op;
	return ThreadScanExclusive<LENGTH>(data, scan_op, seed);
}

/**
 * Serial exclusive scan with the addition operator and seed
 */
template <
	int LENGTH,
	typename T>
__device__ __forceinline__ T ThreadSumExclusive(
	T			(&data)[LENGTH],
	T 			seed)
{
	return ThreadScanExclusive<LENGTH>((T*) data, seed);
}



/******************************************************************************
 * Inclusive scan
 ******************************************************************************/

/**
 * Serial inclusive scan with the specified operator, optionally seeding with
 * a given prefix
 */
template <
	int LENGTH,
	typename T,
	typename ScanOp>
__device__ __forceinline__ T ThreadScanInclusive(
	T* 			data,
	ScanOp 		scan_op,
	T 			seed = data[0],			// Prefix to seed scan with
	bool		apply_seed = true)		// Whether or not the calling thread should apply the seed
{
	// Apply seed if appropriate
	if ((LENGTH > 0) && (apply_seed))
	{
		seed = scan_op(seed, data[0]);
		data[0] = seed;
	}

	// Continue scan
	#pragma unroll
	for (int i = 1; i < LENGTH; ++i)
	{
		seed = scan_op(seed, data[i]);
		data[i] = seed;
	}

	return seed;
}


/**
 * Serial inclusive scan with the addition operator, optionally seeding with
 * a given prefix
 */
template <
	int LENGTH,
	typename T>
__device__ __forceinline__ T ThreadSumInclusive(
	T* 			data,
	T 			seed = data[0],			// Prefix to seed scan with
	bool		apply_seed = true)		// Whether or not the calling thread should apply the seed
{
	Sum<T> scan_op;
	return ThreadScanInclusive<LENGTH>(data, scan_op, seed, apply_seed);
}




} // namespace cub
CUB_NS_POSTFIX

