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
 * Consecutive reduction problem type
 ******************************************************************************/

#pragma once

#include <b40c/scan/problem_type.cuh>
#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace consecutive_reduction {

/**
 * Type of consecutive reduction problem
 */
template <
	typename _KeyType,
	typename _ValueType,
	typename SizeT,
	typename ReductionOp,
	typename IdentityOp,
	typename _EqualityOp>

struct ProblemType :
	scan::ProblemType<_ValueType, SizeT, ReductionOp, IdentityOp, true>		// Inherit from regular scan problem type
{
	// The type of data we are operating upon
	typedef _KeyType 		KeyType;
	typedef _ValueType 		ValueType;
	typedef _EqualityOp		EqualityOp;

	// The size_t type of spine we're using
	typedef int 			SpineSizeT;
};


} // namespace consecutive_reduction
} // namespace b40c

