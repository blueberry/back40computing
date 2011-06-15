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
 * Segmented scan problem type
 ******************************************************************************/

#pragma once

#include <b40c/scan/problem_type.cuh>
#include <b40c/util/soa_tuple.cuh>

namespace b40c {
namespace segmented_scan {

/**
 * Type of segmented scan problem
 */
template <
	typename T,				// Partial type
	typename _Flag,			// Flag type
	typename SizeT,
	bool EXCLUSIVE,
	T BinaryOp(const T&, const T&),
	T Identity()>
struct ProblemType : scan::ProblemType<T, SizeT, EXCLUSIVE, BinaryOp, Identity>		// Inherit from regular scan problem type
{
	// The type of flag we're using
	typedef _Flag Flag;
};


} // namespace segmented_scan
} // namespace b40c

