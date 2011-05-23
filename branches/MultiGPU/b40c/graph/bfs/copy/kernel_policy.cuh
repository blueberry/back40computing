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
 * "Metatype" for guiding BFS copy granularity configuration
 ******************************************************************************/

#pragma once

#include <b40c/copy/policy.cuh>

namespace b40c {
namespace graph {
namespace bfs {
namespace copy {

/**
 * BFS copy kernel policy meta-type.  Parameterizations of this
 * type encapsulate our kernel-tuning parameters (i.e., they are reflected via
 * the static fields).
 *
 * Kernels can be specialized for problem-type, SM-version, etc. by parameterizing
 * them with different performance-tuned parameterizations of this type.  By
 * incorporating this type into the kernel code itself, we guide the compiler in
 * expanding/unrolling the kernel code for specific architectures and problem
 * types.
 */
template <
	// ProblemType type parameters
	typename ProblemType,

	// Machine parameters
	int CUDA_ARCH,

	// Behavioral control parameters
	bool _INSTRUMENT,					// Whether or not we want instrumentation logic generated
	bool _DEQUEUE_PROBLEM_SIZE,			// Whether we obtain problem size from device-side queue counters (true), or use the formal parameter (false)

	// Tunable parameters
	int MAX_CTA_OCCUPANCY,
	int LOG_THREADS,
	int LOG_LOAD_VEC_SIZE,
	int LOG_LOADS_PER_TILE,
	util::io::ld::CacheModifier READ_MODIFIER,
	util::io::st::CacheModifier WRITE_MODIFIER,
	bool WORK_STEALING,
	int LOG_SCHEDULE_GRANULARITY>

struct KernelPolicy :
	b40c::copy::Policy<
		typename ProblemType::VertexId,			// T
		typename ProblemType::SizeT,			// SizeT
		CUDA_ARCH,
		LOG_SCHEDULE_GRANULARITY,
		MAX_CTA_OCCUPANCY,
		LOG_THREADS,
		LOG_LOAD_VEC_SIZE,
		LOG_LOADS_PER_TILE,
		READ_MODIFIER,
		WRITE_MODIFIER,
		WORK_STEALING,
		false>									// OVERSUBSCRIBED_GRID_SIZE
{
	typedef typename ProblemType::VertexId VertexId;


	enum {
		INSTRUMENT						= _INSTRUMENT,
		DEQUEUE_PROBLEM_SIZE			= _DEQUEUE_PROBLEM_SIZE,
	};
};

} // namespace copy
} // namespace bfs
} // namespace graph
} // namespace b40c

