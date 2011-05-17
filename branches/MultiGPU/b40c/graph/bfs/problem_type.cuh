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
 * BFS partition-compaction problem type
 ******************************************************************************/

#pragma once

#include <b40c/partition/problem_type.cuh>
#include <b40c/util/basic_utils.cuh>

namespace b40c {
namespace bfs {


/**
 * Type of BFS problem
 */
template <
	typename 	_VertexId,
	typename 	_SizeT,
	typename 	_CollisionMask,
	typename 	_ValidFlag,
	bool 		_MARK_PARENTS>
struct ProblemType : partition::ProblemType<
	_VertexId, 														// KeyType
	typename If<_MARK_PARENTS, VertexId, util::NullType>::Type,		// ValueType
	_SizeT>															// SizeT
{
	typedef _VertexId														VertexId;
	typedef _CollisionMask													CollisionMask;
	typedef _ValidFlag														ValidFlag;
	static const bool MARK_PARENTS											= _MARK_PARENTS;
};


} // namespace bfs
} // namespace b40c

