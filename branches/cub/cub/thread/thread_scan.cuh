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
 * Exclusive prefix scan
 ******************************************************************************/

/**
 * Exclusive prefix scan across a thread-local array (with specified prefix).
 * Returns the aggregate.
 */
template <
	int LENGTH,					// Length of input/output arrays
	typename T,					// Input/output type
	typename ScanOp>			// Binary scan operator type (parameters of type T)
__device__ __forceinline__ T ThreadScanExclusive(
	T		*input,				// Input array
	T		*output,			// Output array (may be aliased to input)
	ScanOp	scan_op,			// Scan operator
	T 		prefix)				// Prefix to prefix scan with
{
	T exclusive = prefix;
	T inclusive = prefix;

	#pragma unroll
	for (int i = 0; i < LENGTH; ++i)
	{
		inclusive = scan_op(exclusive, input[i]);
		output[i] = (T) exclusive;
		exclusive = inclusive;
	}

	return inclusive;
}


/**
 * Exclusive prefix scan across a thread-local array (with specified seed prefix).
 * Returns the aggregate.
 */
template <
	int LENGTH,					// Length of input/output arrays
	typename T,					// Input/output type
	typename ScanOp>			// Binary scan operator type (parameters of type T)
__device__ __forceinline__ T ThreadScanExclusive(
	T		(&input)[LENGTH],	// Input array
	T		(&output)[LENGTH],	// Output array (may be aliased to input)
	ScanOp	scan_op,			// Scan operator
	T 		prefix)				// Prefix to prefix scan with
{
	return ThreadScanExclusive<LENGTH>((T*) input, (T*) output, scan_op, prefix);
}



/******************************************************************************
 * Inclusive prefix scan
 ******************************************************************************/

/**
 * Inclusive prefix scan across a thread-local array. Returns the aggregate.
 */
template <
	int LENGTH,					// Length of input/output arrays
	typename T,					// Input/output type
	typename ScanOp>			// Scan operator type (functor)
__device__ __forceinline__ T ThreadScanInclusive(
	T		*input,				// Input array
	T		*output,			// Output array (may be aliased to input)
	ScanOp 	scan_op)			// Scan operator
{
	T inclusive = input[0];
	output[0] = inclusive;

	// Continue scan
	#pragma unroll
	for (int i = 0; i < LENGTH; ++i)
	{
		inclusive = scan_op(inclusive, input[i]);
		output[i] = inclusive;
	}

	return inclusive;
}


/**
 * Inclusive prefix scan across a thread-local array (with specified seed prefix).
 * Returns the aggregate.
 */
template <
	int LENGTH,
	typename T,
	typename ScanOp>				// Scan operator type (functor)
__device__ __forceinline__ T ThreadScanInclusive(
	T		*input,					// Input array
	T		*output,				// Output array (may be aliased to input)
	ScanOp 	scan_op,				// Scan operator
	T 		prefix,					// Prefix to prefix scan with (if non-null)
	bool 	apply_prefix = true)	// Whether or not the calling thread should apply its prefix.  (Handy for preventing thread-0 from applying a prefix.)
{
	T inclusive = input[0];

	// Apply prefix if appropriate
	if (apply_prefix)
	{
		inclusive = scan_op(prefix, inclusive);
	}
	output[0] = inclusive;

	// Continue scan
	#pragma unroll
	for (int i = 1; i < LENGTH; ++i)
	{
		inclusive = scan_op(inclusive, input[i]);
		output[i] = inclusive;
	}

	return inclusive;
}





} // namespace cub
CUB_NS_POSTFIX

