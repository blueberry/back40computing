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

#include "b40c_cuda_properties.cuh"
#include "b40c_kernel_data_movement.cuh"

#include "reduction_api_granularity.cuh"
#include "reduction_kernel.cuh"

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
 ******************************************************************************/

/**
 * Default, catch-all granularity parameterization type.  Defers to the
 * architecture "family" that we know we have specialization type(s) for below.
 */
template <
	typename ProblemType,
	int CUDA_ARCH,
	ProblemSize PROBLEM_SIZE,
	int T_SIZE = sizeof(typename ProblemType::T)>
struct TunedConfig : TunedConfig<
	ProblemType,
	FamilyClassifier<CUDA_ARCH>::FAMILY,
	PROBLEM_SIZE,
	T_SIZE> {};


//-----------------------------------------------------------------------------
// SM2.0 specializations(s)
//-----------------------------------------------------------------------------

// Large problems, 8B+ data
template <typename ProblemType, int T_SIZE>
struct TunedConfig<ProblemType, SM20, LARGE, T_SIZE>
	: ReductionConfig<ProblemType, NONE, true, false, true, false, 8, 7, 0, 2, 5, 9, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};
// Large problems, 4B data
template <typename ProblemType>
struct TunedConfig<ProblemType, SM20, LARGE, 4>
	: ReductionConfig<ProblemType, NONE, true, true, false, false, 8, 7, 1, 2, 5, 10, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Large problems, 2B data
template <typename ProblemType>
struct TunedConfig<ProblemType, SM20, LARGE, 2>
	: ReductionConfig<ProblemType, NONE, true, true, false, false, 8, 7, 2, 2, 5, 11, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Large problems, 1B data
template <typename ProblemType>
struct TunedConfig<ProblemType, SM20, LARGE, 1>
	: ReductionConfig<ProblemType, NONE, false, false, false, false, 8, 7, 2, 2, 5, 11, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};



// Small problems
template <typename ProblemType, int T_SIZE>
struct TunedConfig<ProblemType, SM20, SMALL, T_SIZE>
	: ReductionConfig<ProblemType, NONE, false, true, false, false, 8, 5, 2, 1, 5, 8, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = SMALL;
};


//-----------------------------------------------------------------------------
// SM1.3 specializations(s)
//-----------------------------------------------------------------------------

// Large problems
template <typename ProblemType, int T_SIZE>
struct TunedConfig<ProblemType, SM13, LARGE, T_SIZE>
	: ReductionConfig<ProblemType, NONE, false, true, false, false, 8, 7, 1, 2, 5, 10, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = LARGE;
};

// Small problems
template <typename ProblemType, int T_SIZE>
struct TunedConfig<ProblemType, SM13, SMALL, T_SIZE>
	: ReductionConfig<ProblemType, NONE, false, true, false, false, 8, 5, 2, 1, 5, 8, 8, 0, 1, 5>
{
	static const ProblemSize PROBLEM_SIZE = SMALL;
};



//-----------------------------------------------------------------------------
// SM1.0 specializations(s)
//-----------------------------------------------------------------------------


template <ProblemSize _PROBLEM_SIZE, typename ProblemType, int T_SIZE>
struct TunedConfig<ProblemType, SM10, _PROBLEM_SIZE, T_SIZE>
	: ReductionConfig<ProblemType, NONE, false, true, false, false, 8, 7, 1, 2, 5, 10, 8, 0, 1, 5>
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
template <typename ProblemType, int PROBLEM_SIZE>
__launch_bounds__ (
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Upsweep::THREADS),
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Upsweep::CTA_OCCUPANCY))
__global__ void TunedUpsweepReductionKernel(
	typename ProblemType::T 			* __restrict d_in,
	typename ProblemType::T 			* __restrict d_spine,
	typename ProblemType::SizeT 		* __restrict d_work_progress,
	CtaWorkDistribution<typename ProblemType::SizeT> work_decomposition,
	int progress_selector)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::Upsweep Config;

	typename ProblemType::T *d_spine_partial = d_spine + blockIdx.x;

	UpsweepReductionPass<Config, Config::WORK_STEALING>::Invoke(
		d_in,
		d_spine_partial,
		d_work_progress,
		work_decomposition,
		progress_selector);
}


/**
 * Tuned upsweep reduction kernel entry point
 */
template <typename ProblemType, int PROBLEM_SIZE>
__launch_bounds__ (
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Spine::THREADS),
	(TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::ReductionConfig::Spine::CTA_OCCUPANCY))
__global__ void TunedSpineReductionKernel(
	typename ProblemType::T 		* __restrict 	d_spine,
	typename ProblemType::T 		* __restrict 	d_out,
	typename ProblemType::SizeT 					spine_elements)
{
	// Load the tuned granularity type identified by the enum for this architecture
	typedef typename TunedConfig<ProblemType, __B40C_CUDA_ARCH__, (ProblemSize) PROBLEM_SIZE>::Spine Config;

	SpineReductionPass<Config>(d_spine, d_out, spine_elements);
}



}// namespace reduction
}// namespace b40c

