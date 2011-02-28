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
 * Tuned Reduction Granularity Types
 ******************************************************************************/

#pragma once

#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/data_movement_load.cuh>
#include <b40c/util/data_movement_store.cuh>

#include <b40c/reduction/kernel_spine.cuh>
#include <b40c/reduction/kernel_upsweep.cuh>
#include <b40c/reduction/granularity.cuh>

namespace b40c {
namespace reduction {


/******************************************************************************
 * Tuning classifiers to specialize granularity types on
 ******************************************************************************/

/**
 * Enumeration of problem-size genres that we may have tuned for
 */
enum ProblemSize
{
	SMALL 	= 0,
	LARGE 	= 1
};


/**
 * Enumeration of architecture-families that we have tuned for below
 */
enum Family
{
	SM20 	= 200,
	SM13	= 130,
	SM10	= 100
};


/**
 * Classifies a given CUDA_ARCH into an architecture-family
 */
template <int CUDA_ARCH>
struct FamilyClassifier
{
	static const Family FAMILY =	(CUDA_ARCH < SM13) ? 	SM10 :
									(CUDA_ARCH < SM20) ? 	SM13 :
															SM20;
};


/******************************************************************************
 * Granularity tuning types
 *
 * Specialized by family (and optionally by specific architecture) and by
 * problem size type
 *******************************************************************************/

/**
 * Granularity parameterization type
 */
template <
	typename ReductionProblemType,
	int CUDA_ARCH,
	ProblemSize PROBLEM_SIZE,
	typename T = typename ReductionProblemType::T,
	int T_SIZE = sizeof(typename ReductionProblemType::T)>
struct TunedConfig;


/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <typename ReductionProblemType, int CUDA_ARCH, ProblemSize PROBLEM_SIZE, typename T, int T_SIZE>
struct TunedConfig : TunedConfig<
	ReductionProblemType,
	FamilyClassifier<CUDA_ARCH>::FAMILY,
	PROBLEM_SIZE,
	T,
	T_SIZE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 8B+ data
template <typename ReductionProblemType, typename T, int T_SIZE>
struct TunedConfig<ReductionProblemType, SM20, LARGE, T, T_SIZE>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, true, false, true, false, 8, 7, 0, 2, 5, 9, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};
// Large problems, 4B data
template <typename ReductionProblemType, typename T>
struct TunedConfig<ReductionProblemType, SM20, LARGE, T, 4>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, true, true, false, false, 8, 7, 1, 2, 5, 10, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Large problems, 2B data
template <typename ReductionProblemType, typename T>
struct TunedConfig<ReductionProblemType, SM20, LARGE, T, 2>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, true, true, false, false, 8, 7, 2, 2, 5, 11, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Large problems, 1B data
template <typename ReductionProblemType, typename T>
struct TunedConfig<ReductionProblemType, SM20, LARGE, T, 1>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, false, false, false, 8, 7, 2, 2, 5, 11, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};



// Small problems
template <typename ReductionProblemType, typename T, int T_SIZE>
struct TunedConfig<ReductionProblemType, SM20, SMALL, T, T_SIZE>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, true, false, false, 8, 5, 2, 1, 5, 8, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = SMALL;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 4B+
template <typename ReductionProblemType, typename T, int T_SIZE>
struct TunedConfig<ReductionProblemType, SM13, LARGE, T, T_SIZE>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, false, false, false, 8, 6, 0, 2, 5, 8, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Large problems, 2B
template <typename ReductionProblemType, typename T>
struct TunedConfig<ReductionProblemType, SM13, LARGE, T, 2>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, false, false, false, 8, 6, 1, 2, 5, 9, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Large problems, 1B
template <typename ReductionProblemType, typename T>
struct TunedConfig<ReductionProblemType, SM13, LARGE, T, 1>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, false, false, false, 4, 8, 2, 2, 5, 12, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Small problems
template <typename ReductionProblemType, typename T, int T_SIZE>
struct TunedConfig<ReductionProblemType, SM13, SMALL, T, T_SIZE>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, false, false, false, 8, 5, 0, 2, 5, 7, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = SMALL;
};



//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------


template <typename ReductionProblemType, ProblemSize _PROBLEM_SIZE, typename T, int T_SIZE>
struct TunedConfig<ReductionProblemType, SM10, _PROBLEM_SIZE, T, T_SIZE>
	: ReductionConfig<ReductionProblemType, util::ld::NONE, util::st::NONE, false, false, false, false, 8, 6, 0, 2, 5, 8, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = _PROBLEM_SIZE;
};





/******************************************************************************
 * Reduction kernel entry points that can derive a tuned granularity type
 * implicitly from the PROBLEM_SIZE template parameter.  (As opposed to having
 * the granularity type passed explicitly.)
 *
 * TODO: Section can be removed if CUDA Runtime is fixed to
 * properly support template specialization around kernel call sites.
 ******************************************************************************/

/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ReductionProblemType, int PROBLEM_SIZE>
__launch_bounds__ (
	(TunedConfig<ReductionProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Upsweep::THREADS),
	(TunedConfig<ReductionProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Upsweep::CTA_OCCUPANCY))
__global__ void TunedUpsweepReductionKernel(
	typename ReductionProblemType::T 			* __restrict d_in,
	typename ReductionProblemType::T 			* __restrict d_spine,
	typename ReductionProblemType::SizeT 		* __restrict d_work_progress,
	util::CtaWorkDistribution<typename ReductionProblemType::SizeT> work_decomposition,
	int progress_selector)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ReductionProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::Upsweep ReductionKernelConfig;

	typename ReductionProblemType::T *d_spine_partial = d_spine + blockIdx.x;

	UpsweepReductionPass<ReductionKernelConfig, ReductionKernelConfig::WORK_STEALING>::Invoke(
		d_in,
		d_spine_partial,
		d_work_progress,
		work_decomposition,
		progress_selector);
}


/**
 * Tuned spine reduction kernel entry point
 */
template <typename ReductionProblemType, int PROBLEM_SIZE>
__launch_bounds__ (
	(TunedConfig<ReductionProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Spine::THREADS),
	(TunedConfig<ReductionProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Spine::CTA_OCCUPANCY))
__global__ void TunedSpineReductionKernel(
	typename ReductionProblemType::T 		* __restrict 	d_spine,
	typename ReductionProblemType::T 		* __restrict 	d_out,
	typename ReductionProblemType::SizeT 					spine_elements)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ReductionProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::Spine ReductionKernelConfig;

	SpineReductionPass<ReductionKernelConfig>(d_spine, d_out, spine_elements);
}



}// namespace reduction
}// namespace b40c

