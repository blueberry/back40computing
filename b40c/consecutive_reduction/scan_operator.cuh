/******************************************************************************
 * 
 * Copyright 2010-2011 Duane Merrill
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
 ******************************************************************************/

/******************************************************************************
 * Scan operator for consecutive reduction problems
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace consecutive_reduction {

/**
 * Structure-of-array scan operator
 */
template <
	typename ReductionOp,	// Type of reduction operator for values
	typename IdentityOp,	// Type of identity operator for values
	typename SoaTuple>		// SOA-tuple type of (value, flag)
struct SoaScanOp
{
	typedef typename SoaTuple::T0 	ValueType;
	typedef typename SoaTuple::T1 	FlagType;

	enum {
		TRANSLATE_FLAG_IDENTITY = 0,	// The flag identity does not need to be translated
	};

	// Operators instances
	ReductionOp 		reduction_op;
	IdentityOp 			identity_op;

	// Constructor
	__device__ __forceinline__ SoaScanOp(
		ReductionOp reduction_op,
		IdentityOp identity_op) :
			reduction_op(reduction_op),
			identity_op(identity_op)
	{}

	// SOA scan operator
	__device__ __forceinline__ SoaTuple operator()(
		const SoaTuple &first,
		const SoaTuple &second)
	{
		if (second.t1) {
			return SoaTuple(second.t0, first.t1 + second.t1);
		} else {
			return SoaTuple(reduction_op(first.t0, second.t0), first.t1 + second.t1);
		}
	}

	// Flag identity
	static __device__ __forceinline__ FlagType FlagIdentity()
	{
		return 0;
	}

	// SOA identity operator
	__device__ __forceinline__ SoaTuple operator()()
	{
		return SoaTuple(
			identity_op(),
			FlagIdentity());
	}
};


/**
 * Structure-of-array scan operator, specialized for problems without
 * an identity operator for values.
 *
 * This essentially works by creating our own "identity" tuples
 * that can't be reduced with others.
 */
template <
	typename ReductionOp,
	typename SoaTuple>
struct SoaScanOp<ReductionOp, util::NullType, SoaTuple>
{
	typedef typename SoaTuple::T0 	ValueType;
	typedef typename SoaTuple::T1 	FlagType;

	enum {
		TRANSLATE_FLAG_IDENTITY = 1,		// The flag identity must be translated from -1 to 0
	};

	// Operators instances
	ReductionOp 		reduction_op;

	// Constructor
	__device__ __forceinline__ SoaScanOp(
		ReductionOp reduction_op,
		util::NullType identity_op) :
			reduction_op(reduction_op)
	{}

	// SOA scan operator
	__device__ __forceinline__ SoaTuple operator()(
		const SoaTuple &first,
		const SoaTuple &second)
	{
		if (second.t1 < 0) {
			return first;					// second is an invalid identity tuple
		} else if (first.t1 < 0) {
			return second;					// first is an invalid identity tuple
		} else if (second.t1) {
			return SoaTuple(second.t0, first.t1 + second.t1);
		} else {
			return SoaTuple(reduction_op(first.t0, second.t0), first.t1 + second.t1);
		}
	}

	// Flag identity
	static __device__ __forceinline__ FlagType FlagIdentity()
	{
		return -1;
	}

	// SOA identity operator
	__device__ __forceinline__ SoaTuple operator()()
	{
		SoaTuple retval;
		retval.t1 = FlagIdentity();
		return retval;
	}
};


} // namespace consecutive_reduction
} // namespace b40c

